use std::{collections::HashSet, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;
use goose_agent::{
    machine::{StateMachine, Step},
    operation::{messages_since_kickoff, not_applicable, Emitter, Operation, OperationResult},
    tool::ToolOperation,
};
use goose_provider_types::conversation::{
    message::{Message, MessageContent},
    Conversation,
};
use rmcp::model::{CallToolResult, ContentBlock};
use tokio_util::sync::CancellationToken;

use crate::agent::{messages::ConversationMessage, progress::AgentProgress};

use super::{
    convert::{ensure_nonempty_kickoff, final_text, goose_to_openhuman, openhuman_to_goose},
    inference::OpenHumanInference,
    store::{rebuild_action_index, CheckpointRuntime},
    tools::{GooseToolSecurity, OpenHumanToolProvider},
    types::{GooseCheckpoint, GooseSession, GooseStopReason, GooseTurnOutcome, OpenHumanEffect},
    GooseCheckpointStore,
};

struct CancellationObservation;

#[async_trait]
impl Operation<GooseSession, OpenHumanEffect> for CancellationObservation {
    fn name(&self) -> &'static str {
        "openhuman_cancellation"
    }

    async fn run(
        &self,
        _session: &GooseSession,
        _conversation: &Conversation,
        _emit: &Emitter,
    ) -> Result<OperationResult<OpenHumanEffect>> {
        not_applicable()
    }

    async fn cancel(
        &self,
        _session: &GooseSession,
        conversation: &Conversation,
        result: OperationResult<OpenHumanEffect>,
        emit: &Emitter,
    ) -> Result<OperationResult<OpenHumanEffect>> {
        if !matches!(result, OperationResult::NotApplicable) {
            return Ok(result);
        }
        let turn = messages_since_kickoff(conversation)?;
        let answered: HashSet<&str> = turn
            .iter()
            .flat_map(Message::get_tool_response_ids)
            .collect();
        let pending = turn
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(MessageContent::as_tool_request)
            .filter(|request| !answered.contains(request.id.as_str()))
            .collect::<Vec<_>>();
        if pending.is_empty() {
            return goose_agent::operation::yielded();
        }
        let mut response = Message::user();
        for request in pending {
            response = response.with_tool_response(
                request.id.clone(),
                Ok(CallToolResult::error(vec![ContentBlock::text(
                    "Tool call was cancelled before execution",
                )])),
            );
        }
        let response = emit.message(response).await;
        goose_agent::operation::yielded_with([OpenHumanEffect::from(response)])
    }
}

/// Direct adapter around the vendored `goose_agent::machine::StateMachine`.
/// It is dormant until a later gate selects it for a primary turn, so the
/// existing TinyAgents dispatch path remains unchanged.
pub struct GooseTurnAdapter {
    pub store: Arc<dyn GooseCheckpointStore>,
    pub model: Arc<dyn tinyinference::model::ChatModel<()>>,
    pub model_name: String,
    pub provider_id: String,
    pub tools: Vec<Arc<dyn crate::tools::Tool>>,
    pub security: Arc<dyn GooseToolSecurity>,
    pub progress: Option<tokio::sync::mpsc::Sender<AgentProgress>>,
    pub cancel: CancellationToken,
    pub max_output_tokens: Option<u32>,
}

impl GooseTurnAdapter {
    pub fn checkpoint_from_openhuman(messages: &[ConversationMessage]) -> Result<GooseCheckpoint> {
        let conversation = openhuman_to_goose(messages)?;
        ensure_nonempty_kickoff(&conversation)?;
        let mut checkpoint = GooseCheckpoint::new(conversation);
        rebuild_action_index(&mut checkpoint)?;
        Ok(checkpoint)
    }

    async fn progress(&self, event: AgentProgress) {
        if let Some(progress) = &self.progress {
            let _ = progress.send(event).await;
        }
    }

    pub async fn run(&self, session_id: &str) -> Result<GooseTurnOutcome> {
        self.progress(AgentProgress::TurnStarted).await;

        let provider = Arc::new(OpenHumanToolProvider {
            tools: self.tools.clone(),
            security: self.security.clone(),
            store: self.store.clone(),
            progress: self.progress.clone(),
        });
        let tools = ToolOperation::<GooseSession>::new().with_provider(provider);
        let inference = OpenHumanInference {
            model: self.model.clone(),
            model_name: self.model_name.clone(),
            provider_id: self.provider_id.clone(),
            max_output_tokens: self.max_output_tokens,
            progress: self.progress.clone(),
        };
        let machine = StateMachine::new(
            vec![
                Step::Operation(Arc::new(CancellationObservation)),
                Step::Operation(Arc::new(tools)),
                Step::Inference(Arc::new(inference)),
            ],
            self.cancel.clone(),
        );
        let runtime = CheckpointRuntime {
            store: self.store.clone(),
        };
        let (events_tx, mut events_rx) = tokio::sync::mpsc::channel(64);
        let emit = Emitter::new(events_tx, self.cancel.clone());
        let drain = tokio::spawn(async move { while events_rx.recv().await.is_some() {} });
        let session = machine.run(&runtime, session_id, &emit).await?;
        drop(emit);
        let _ = drain.await;

        let final_answer = final_text(&session.checkpoint.conversation);
        let stop_reason = if self.cancel.is_cancelled() {
            GooseStopReason::Cancelled
        } else if final_answer.is_some() {
            GooseStopReason::FinalAnswer
        } else {
            GooseStopReason::Yielded
        };
        if let Some(answer) = final_answer {
            self.progress(AgentProgress::TurnContent {
                input: None,
                output: Some(answer),
            })
            .await;
            self.progress(AgentProgress::TurnCompleted {
                iterations: session.checkpoint.usage.primary_calls,
            })
            .await;
        }
        Ok(GooseTurnOutcome {
            openhuman_messages: goose_to_openhuman(&session.checkpoint.conversation),
            checkpoint: session.checkpoint,
            stop_reason,
        })
    }
}
