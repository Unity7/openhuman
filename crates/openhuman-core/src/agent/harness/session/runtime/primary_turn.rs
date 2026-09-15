//! Phase 7 primary interactive entrypoint.
//!
//! It deliberately runs before `Agent::turn`: direct chat therefore cannot
//! trigger the pre-turn memory, goal, skill, or delegation preparation owned by
//! that legacy path.  TinyAgents remains available as the rollback branch.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use tinyinference::{
    message::Message,
    model::{ModelRequest, ToolChoice},
};

use super::super::types::Agent;
use crate::{
    agent::{
        goose::{GooseCheckpointStore, GooseStopReason, GooseTurnAdapter},
        messages::{ChatMessage, ConversationMessage},
        primary_orchestration::{primary_checkpoint_store, PrimaryTurnMode},
        progress::AgentProgress,
    },
    config::OrchestrationEngine,
};

const DIRECT_CHAT_SYSTEM: &str = concat!(
    include_str!("../../../prompts/IDENTITY.md"),
    "\n\n",
    include_str!("../../../prompts/STYLE.md"),
    "\n\n## Chat mode\nAnswer from the supplied conversation and your model knowledge. ",
    "Do not use or claim to use tools, memory, goals, skills, delegation, external services, ",
    "or current-data retrieval. If live information is required, say that an assisted turn is needed."
);

impl Agent {
    /// Execute a primary interactive turn using the already-resolved session
    /// model.  This is the sole web-chat branch added by the mode gate.
    pub(crate) async fn run_primary_interactive(
        &mut self,
        message: &str,
        mode: PrimaryTurnMode,
        engine: OrchestrationEngine,
        checkpoint_id: &str,
    ) -> Result<String> {
        if engine == OrchestrationEngine::Tinyagents {
            return self.run_single(message).await;
        }

        let history_snapshot = self.begin_guarded_run(message)?;
        let result = match mode {
            PrimaryTurnMode::Chat => self.run_direct_chat(message).await,
            PrimaryTurnMode::Assist | PrimaryTurnMode::Agent => {
                self.run_goose_primary(message, mode, checkpoint_id).await
            }
        };
        self.finish_guarded_run(&history_snapshot, result)
    }

    async fn resolved_primary_model(
        &self,
    ) -> Result<(crate::agent::tinyagents::TurnModels, Option<u64>)> {
        let context_window = self
            .turn_model_source
            .effective_context_window(&self.model_name)
            .await;
        let models =
            self.turn_model_source
                .build(&self.model_name, self.temperature, context_window)?;
        Ok((models, context_window))
    }

    fn direct_messages(&mut self, user_message: &str) -> Vec<Message> {
        self.absorb_resumed_transcript_prefix();
        let mut messages = vec![Message::system(DIRECT_CHAT_SYSTEM)];
        messages.extend(
            self.tool_dispatcher
                .to_provider_messages(&self.history)
                .into_iter()
                .filter(|message| message.role != "system")
                .map(|message| crate::agent::tinyagents::chat_message_to_message(&message)),
        );
        messages.push(Message::user(user_message.to_string()));
        messages
    }

    async fn run_direct_chat(&mut self, user_message: &str) -> Result<String> {
        let started = std::time::Instant::now();
        let (models, context_window) = self.resolved_primary_model().await?;
        let request = ModelRequest::new(self.direct_messages(user_message))
            .with_model(self.model_name.clone())
            .with_temperature(self.temperature)
            .with_tool_choice(ToolChoice::None);

        self.emit_primary_progress(AgentProgress::TurnStarted).await;
        self.emit_primary_progress(AgentProgress::IterationStarted {
            iteration: 1,
            max_iterations: 1,
        })
        .await;
        let response = models.primary.invoke(&(), request).await?;
        let reply = response.text();
        if reply.trim().is_empty() {
            return Err(anyhow!("The model returned an empty response."));
        }
        let usage = response
            .usage
            .or(response.message.usage)
            .unwrap_or_default();
        tracing::info!(
            mode = "chat",
            stop_reason = "final_answer",
            context_window = context_window.unwrap_or(0),
            latest_context_occupancy = usage.input_tokens,
            cumulative_input_tokens = usage.input_tokens,
            cumulative_output_tokens = usage.output_tokens,
            primary_calls = 1,
            advertised_tools = 0,
            "[primary-orchestration] direct chat stopped"
        );

        self.history
            .push(ConversationMessage::Chat(ChatMessage::user(
                user_message.to_string(),
            )));
        self.history
            .push(ConversationMessage::Chat(ChatMessage::assistant(
                reply.clone(),
            )));
        self.finish_primary_mode_turn(
            user_message,
            &reply,
            context_window,
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_tokens,
            1,
            started,
            true,
        )
        .await;
        Ok(reply)
    }

    async fn run_goose_primary(
        &mut self,
        user_message: &str,
        mode: PrimaryTurnMode,
        checkpoint_id: &str,
    ) -> Result<String> {
        let started = std::time::Instant::now();
        let (models, context_window) = self.resolved_primary_model().await?;
        self.absorb_resumed_transcript_prefix();

        // Gate 3 proves dispatch through Goose while exposing no task tools.
        // Gate 4 supplies the typed ordered capability plan and security-backed
        // tool set.  An empty registry fails closed and cannot execute a tool.
        let mut messages = Vec::with_capacity(self.history.len() + 2);
        messages.push(ConversationMessage::Chat(ChatMessage::system(
            "You are OpenHuman. Complete the user's request directly. If a required capability is unavailable, explain that clearly without claiming it ran.",
        )));
        messages.extend(self.history.iter().filter(|entry| {
            !matches!(entry, ConversationMessage::Chat(chat) if chat.role == "system")
        }).cloned());
        messages.push(ConversationMessage::Chat(ChatMessage::user(
            user_message.to_string(),
        )));

        let store = primary_checkpoint_store(&self.workspace_dir);
        let resume = mode == PrimaryTurnMode::Agent
            && GooseCheckpointStore::load(store.as_ref(), checkpoint_id)
                .await
                .is_ok_and(|checkpoint| checkpoint.is_resumable());
        if !resume {
            let checkpoint = GooseTurnAdapter::checkpoint_from_openhuman(&messages)?;
            store.insert(checkpoint_id, checkpoint)?;
        }
        let store_dyn: Arc<dyn GooseCheckpointStore> = store;

        let runtime = self.runtime_config.clone().unwrap_or_default();
        let policy = Arc::new(crate::security::SecurityPolicy::from_config(
            &runtime.autonomy,
            &runtime.workspace_dir,
            &runtime.action_dir,
        ));
        let security = Arc::new(crate::agent::tinyagents::host::OpenHumanSecurityGate::new(
            policy,
            Vec::new(),
        ));
        let provider_id = models.provider_id().to_string();
        let adapter = GooseTurnAdapter {
            store: store_dyn,
            model: models.primary,
            model_name: self.model_name.clone(),
            provider_id,
            tools: Vec::new(),
            security,
            progress: self.on_progress.clone(),
            cancel: tokio_util::sync::CancellationToken::new(),
            max_output_tokens: None,
            max_primary_calls: mode.max_primary_calls(),
        };
        let outcome = adapter.run(checkpoint_id).await?;
        tracing::info!(
            mode = mode.as_str(),
            stop_reason = outcome.stop_reason.as_str(),
            context_window = context_window.unwrap_or(0),
            latest_context_occupancy = outcome.checkpoint.usage.latest_primary_input_tokens,
            cumulative_input_tokens = outcome.checkpoint.usage.cumulative_input_tokens,
            cumulative_output_tokens = outcome.checkpoint.usage.cumulative_output_tokens,
            primary_calls = outcome.checkpoint.usage.primary_calls,
            "[primary-orchestration] Goose turn stopped"
        );
        let reply = outcome
            .openhuman_messages
            .iter()
            .rev()
            .find_map(|entry| match entry {
                ConversationMessage::Chat(chat)
                    if chat.role == "assistant" && !chat.content.trim().is_empty() =>
                {
                    Some(chat.content.clone())
                }
                _ => None,
            })
            .ok_or_else(|| match outcome.stop_reason {
                GooseStopReason::Cancelled => anyhow!("The turn was cancelled."),
                GooseStopReason::CallCeiling if mode == PrimaryTurnMode::Agent => anyhow!(
                    "The autonomous turn reached its call ceiling and was checkpointed for resume."
                ),
                GooseStopReason::CallCeiling => anyhow!(
                    "The assisted turn reached its call ceiling before producing an answer."
                ),
                GooseStopReason::Yielded => anyhow!("The turn paused before producing an answer."),
                GooseStopReason::FinalAnswer => anyhow!("The model returned an empty response."),
            })?;

        self.history
            .push(ConversationMessage::Chat(ChatMessage::user(
                user_message.to_string(),
            )));
        self.history
            .extend(outcome.openhuman_messages.into_iter().skip(messages.len()));
        if !matches!(self.history.last(), Some(ConversationMessage::Chat(chat)) if chat.role == "assistant" && chat.content == reply)
        {
            self.history
                .push(ConversationMessage::Chat(ChatMessage::assistant(
                    reply.clone(),
                )));
        }

        let usage = outcome.checkpoint.usage;
        self.finish_primary_mode_turn(
            user_message,
            &reply,
            context_window,
            usage.cumulative_input_tokens,
            usage.cumulative_output_tokens,
            usage.cumulative_cached_input_tokens,
            usage.primary_calls,
            started,
            false,
        )
        .await;
        Ok(reply)
    }

    #[allow(clippy::too_many_arguments)]
    async fn finish_primary_mode_turn(
        &mut self,
        user_message: &str,
        reply: &str,
        context_window: Option<u64>,
        input_tokens: u64,
        output_tokens: u64,
        cached_input_tokens: u64,
        iterations: u32,
        started: std::time::Instant,
        emit_terminal_progress: bool,
    ) {
        self.trim_history();
        self.last_turn_hit_cap = false;
        self.last_turn_citations.clear();
        self.pending_citations = None;
        self.last_memory_context = None;
        self.last_turn_usage_totals =
            Some(crate::agent::harness::turn_subagent_usage::LastTurnUsage {
                input_tokens,
                output_tokens,
                cached_input_tokens,
                cost_usd: 0.0,
                context_window: context_window.unwrap_or(0),
                context_used_tokens: input_tokens.saturating_add(output_tokens),
                subagents: Vec::new(),
            });

        let persisted = self.tool_dispatcher.to_provider_messages(&self.history);
        let turn_usage = crate::agent::harness::session::transcript::TurnUsage {
            provider: self.event_channel.clone(),
            model: self.model_name.clone(),
            usage: crate::agent::harness::session::transcript::MessageUsage {
                input: input_tokens,
                output: output_tokens,
                cached_input: cached_input_tokens,
                context_window: context_window.unwrap_or(0),
                cost_usd: 0.0,
            },
            ts: chrono::Utc::now().to_rfc3339(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            iteration: iterations,
        };
        self.persist_session_transcript(
            &persisted,
            input_tokens,
            output_tokens,
            cached_input_tokens,
            0.0,
            Some(&turn_usage),
        );

        let capture_content = self
            .runtime_config
            .as_ref()
            .map(|config| config.observability.agent_tracing.capture_content)
            .unwrap_or(false);
        if emit_terminal_progress && capture_content {
            self.emit_primary_progress(AgentProgress::TurnContent {
                input: Some(user_message.to_string()),
                output: Some(reply.to_string()),
            })
            .await;
        }
        if emit_terminal_progress {
            self.emit_primary_progress(AgentProgress::TurnCompleted { iterations })
                .await;
        }

        if !self.post_turn_hooks.is_empty() {
            crate::agent::hooks::fire_hooks(
                &self.post_turn_hooks,
                crate::agent::hooks::TurnContext {
                    user_message: user_message.to_string(),
                    assistant_response: reply.to_string(),
                    tool_calls: Vec::new(),
                    turn_duration_ms: started.elapsed().as_millis() as u64,
                    session_id: Some(self.event_session_id.clone())
                        .filter(|value| !value.trim().is_empty()),
                    agent_id: Some(self.agent_definition_id.clone())
                        .filter(|value| !value.trim().is_empty()),
                    entrypoint: Some(self.event_channel.clone())
                        .filter(|value| !value.trim().is_empty()),
                    iteration_count: iterations as usize,
                },
            );
        }
    }

    async fn emit_primary_progress(&self, event: AgentProgress) {
        if let Some(progress) = &self.on_progress {
            let _ = progress.send(event).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use parking_lot::Mutex;
    use tinyinference::model::{
        ChatModel, ModelProfile, ModelRequest, ModelResponse, ModelStream, ModelStreamItem,
    };

    use super::*;

    struct NeverCalledTool;

    #[async_trait]
    impl crate::tools::Tool for NeverCalledTool {
        fn name(&self) -> &str {
            "never_called"
        }

        fn description(&self) -> &str {
            "Test-only tool that must not be exposed by the mode gate."
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
        ) -> anyhow::Result<crate::tools::ToolResult> {
            panic!("mode-gate test tool must never execute")
        }
    }

    #[derive(Default)]
    struct RecordingModel {
        requests: Mutex<Vec<ModelRequest>>,
    }

    #[async_trait]
    impl ChatModel<()> for RecordingModel {
        fn profile(&self) -> Option<&ModelProfile> {
            static PROFILE: std::sync::LazyLock<ModelProfile> =
                std::sync::LazyLock::new(ModelProfile::default);
            Some(&PROFILE)
        }

        async fn invoke(
            &self,
            _state: &(),
            request: ModelRequest,
        ) -> tinyinference::Result<ModelResponse> {
            self.requests.lock().push(request);
            Ok(ModelResponse::assistant("mock answer"))
        }

        async fn stream(
            &self,
            state: &(),
            request: ModelRequest,
        ) -> tinyinference::Result<ModelStream> {
            let response = self.invoke(state, request).await?;
            Ok(Box::pin(futures::stream::iter(vec![
                ModelStreamItem::Started,
                ModelStreamItem::Completed(response),
            ])))
        }
    }

    fn agent(model: Arc<RecordingModel>, workspace: &std::path::Path) -> Agent {
        Agent::builder()
            .chat_model(model)
            .tools(vec![Box::new(NeverCalledTool)])
            .memory(crate::memory::test_support::noop_memory())
            .tool_dispatcher(Box::new(crate::agent::dispatcher::NativeToolDispatcher))
            .workspace_dir(workspace.to_path_buf())
            .model_name("qwen38-openhuman".to_string())
            .build()
            .expect("build test agent")
    }

    #[tokio::test]
    async fn greeting_and_general_question_each_make_one_call_with_zero_tools() {
        for prompt in ["hey", "explain why the sky is blue"] {
            let workspace = tempfile::tempdir().expect("temp workspace");
            let model = Arc::new(RecordingModel::default());
            let mut agent = agent(model.clone(), workspace.path());

            let answer = agent
                .run_primary_interactive(
                    prompt,
                    PrimaryTurnMode::Chat,
                    OrchestrationEngine::Goose,
                    "chat-test",
                )
                .await
                .expect("direct chat succeeds");
            assert_eq!(answer, "mock answer");
            let requests = model.requests.lock();
            assert_eq!(requests.len(), 1, "{prompt}");
            assert!(requests[0].tools.is_empty(), "{prompt}");
            assert_eq!(requests[0].tool_choice, ToolChoice::None, "{prompt}");
            assert!(requests[0]
                .messages
                .iter()
                .any(|message| message.text().contains("## Chat mode")));
            assert!(!requests[0]
                .messages
                .iter()
                .any(|message| message.text().contains("active_goal")));
        }
    }

    #[tokio::test]
    async fn assist_and_agent_dispatch_through_goose_with_no_gate4_tools() {
        for mode in [PrimaryTurnMode::Assist, PrimaryTurnMode::Agent] {
            let workspace = tempfile::tempdir().expect("temp workspace");
            let model = Arc::new(RecordingModel::default());
            let mut agent = agent(model.clone(), workspace.path());

            let answer = agent
                .run_primary_interactive(
                    "perform the requested action",
                    mode,
                    OrchestrationEngine::Goose,
                    mode.as_str(),
                )
                .await
                .expect("Goose turn succeeds");
            assert_eq!(answer, "mock answer");
            let requests = model.requests.lock();
            assert_eq!(requests.len(), 1, "mode={}", mode.as_str());
            assert!(requests[0].tools.is_empty());
            assert!(requests[0]
                .messages
                .iter()
                .any(|message| message.text().contains("Complete the user's request")));
        }
    }

    #[test]
    fn tinyagents_setting_keeps_the_rollback_path() {
        std::thread::Builder::new()
            .name("phase7-tinyagents-rollback".to_string())
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("test runtime")
                    .block_on(async {
                        let workspace = tempfile::tempdir().expect("temp workspace");
                        let model = Arc::new(RecordingModel::default());
                        let mut agent = agent(model.clone(), workspace.path());

                        let answer = agent
                            .run_primary_interactive(
                                "hey",
                                PrimaryTurnMode::Chat,
                                OrchestrationEngine::Tinyagents,
                                "rollback",
                            )
                            .await
                            .expect("TinyAgents rollback succeeds");
                        assert_eq!(answer, "mock answer");
                        assert_eq!(model.requests.lock().len(), 1);
                    });
            })
            .expect("spawn rollback test")
            .join()
            .expect("rollback test thread");
    }
}
