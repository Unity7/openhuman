use std::{
    collections::{HashMap, HashSet},
    sync::Mutex,
};

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use goose_agent::{
    machine::{EffectHandler, SessionLoader},
    operation::ConversationEffect,
};
use goose_provider_types::conversation::message::MessageContent;

use super::types::{
    AcceptedToolAction, GooseCheckpoint, GooseSession, OpenHumanEffect, ToolObservation,
};

/// Optimistic checkpoint store used by the adapter. Implementations must make
/// `compare_and_swap` atomic: every Goose step is one durable commit.
#[async_trait]
pub trait GooseCheckpointStore: Send + Sync {
    async fn load(&self, session_id: &str) -> Result<GooseCheckpoint>;

    async fn compare_and_swap(
        &self,
        session_id: &str,
        expected_revision: u64,
        checkpoint: GooseCheckpoint,
    ) -> Result<()>;

    /// Durably claim one accepted action before its side effect starts.
    /// Returns `false` when a prior process already claimed it, in which case
    /// the adapter must not execute it again.
    async fn claim_execution(&self, session_id: &str, call_id: &str) -> Result<bool>;
}

/// Network-free store suitable for tests and embedders that supply their own
/// durable store later. It still enforces the exact optimistic-write contract.
#[derive(Default)]
pub struct InMemoryGooseCheckpointStore {
    checkpoints: Mutex<HashMap<String, GooseCheckpoint>>,
    execution_claims: Mutex<HashSet<(String, String)>>,
}

impl InMemoryGooseCheckpointStore {
    pub fn insert(&self, session_id: impl Into<String>, checkpoint: GooseCheckpoint) {
        self.checkpoints
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(session_id.into(), checkpoint);
    }
}

#[async_trait]
impl GooseCheckpointStore for InMemoryGooseCheckpointStore {
    async fn load(&self, session_id: &str) -> Result<GooseCheckpoint> {
        self.checkpoints
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(session_id)
            .cloned()
            .ok_or_else(|| anyhow!("Goose checkpoint '{session_id}' does not exist"))
    }

    async fn compare_and_swap(
        &self,
        session_id: &str,
        expected_revision: u64,
        checkpoint: GooseCheckpoint,
    ) -> Result<()> {
        let mut checkpoints = self
            .checkpoints
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let current = checkpoints
            .get(session_id)
            .ok_or_else(|| anyhow!("Goose checkpoint '{session_id}' does not exist"))?;
        if current.revision != expected_revision {
            return Err(anyhow!(
                "stale Goose checkpoint: expected revision {expected_revision}, found {}",
                current.revision
            ));
        }
        checkpoints.insert(session_id.to_string(), checkpoint);
        Ok(())
    }

    async fn claim_execution(&self, session_id: &str, call_id: &str) -> Result<bool> {
        Ok(self
            .execution_claims
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert((session_id.to_string(), call_id.to_string())))
    }
}

pub(super) struct CheckpointRuntime {
    pub store: std::sync::Arc<dyn GooseCheckpointStore>,
}

#[async_trait]
impl SessionLoader<GooseSession> for CheckpointRuntime {
    async fn load(&self, session_id: &str) -> Result<GooseSession> {
        Ok(GooseSession {
            id: session_id.to_string(),
            checkpoint: self.store.load(session_id).await?,
        })
    }
}

fn text_from_result(result: &rmcp::model::CallToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|block| block.as_text().map(|text| text.text.as_str()))
        .collect::<Vec<_>>()
        .join("\n")
}

fn index_message(
    checkpoint: &mut GooseCheckpoint,
    message: &goose_provider_types::conversation::message::Message,
) -> Result<()> {
    for content in &message.content {
        match content {
            MessageContent::ToolRequest(request) => {
                let call = request.tool_call.as_ref().map_err(|error| {
                    anyhow!(
                        "cannot persist malformed accepted tool action {}: {error}",
                        request.id
                    )
                })?;
                let arguments =
                    serde_json::Value::Object(call.arguments.clone().unwrap_or_default());
                let action = AcceptedToolAction {
                    call_id: request.id.clone(),
                    tool_name: call.name.to_string(),
                    arguments,
                    observation: None,
                };
                if checkpoint.actions.contains_key(&request.id) {
                    return Err(anyhow!(
                        "tool call id '{}' was persisted more than once",
                        request.id
                    ));
                }
                checkpoint.actions.insert(request.id.clone(), action);
            }
            MessageContent::ToolResponse(response) => {
                let action = checkpoint.actions.get_mut(&response.id).ok_or_else(|| {
                    anyhow!("orphaned tool observation for call '{}'", response.id)
                })?;
                let (success, output) = match &response.tool_result {
                    Ok(result) => (!result.is_error.unwrap_or(false), text_from_result(result)),
                    Err(error) => (false, error.message.to_string()),
                };
                let observation = ToolObservation {
                    call_id: response.id.clone(),
                    success,
                    output,
                };
                if action.observation.is_some() {
                    return Err(anyhow!(
                        "tool call '{}' received more than one observation",
                        response.id
                    ));
                }
                action.observation = Some(observation);
            }
            _ => {}
        }
    }
    Ok(())
}

pub(super) fn rebuild_action_index(checkpoint: &mut GooseCheckpoint) -> Result<()> {
    checkpoint.actions.clear();
    let messages = checkpoint.conversation.messages().clone();
    for message in &messages {
        index_message(checkpoint, message)?;
    }
    Ok(())
}

fn apply_conversation_effect(
    checkpoint: &mut GooseCheckpoint,
    effect: ConversationEffect,
) -> Result<()> {
    match effect {
        ConversationEffect::AppendMessage(message) => {
            index_message(checkpoint, &message)?;
            checkpoint.conversation.push(message);
        }
        ConversationEffect::ReplaceConversation(conversation) => {
            let mut rebuilt = GooseCheckpoint::new(conversation.clone());
            rebuilt.revision = checkpoint.revision;
            rebuilt.usage = checkpoint.usage.clone();
            for message in conversation.messages() {
                index_message(&mut rebuilt, message)?;
            }
            checkpoint.conversation = conversation;
            checkpoint.actions = rebuilt.actions;
        }
        ConversationEffect::PatchToolRequestMeta {
            tool_call_id,
            patch,
        } => {
            let request = checkpoint
                .conversation
                .messages_mut()
                .iter_mut()
                .flat_map(|message| message.content.iter_mut())
                .filter_map(|content| match content {
                    MessageContent::ToolRequest(request) if request.id == tool_call_id => {
                        Some(request)
                    }
                    _ => None,
                })
                .next()
                .ok_or_else(|| {
                    anyhow!("tool request '{tool_call_id}' not found for metadata patch")
                })?;
            request.tool_meta = Some(patch);
        }
        ConversationEffect::SetMessageVisibility {
            message_id,
            user_visible,
            agent_visible,
        } => {
            let message = checkpoint
                .conversation
                .messages_mut()
                .iter_mut()
                .find(|message| message.id.as_deref() == Some(&message_id))
                .ok_or_else(|| anyhow!("message '{message_id}' not found for visibility update"))?;
            message.metadata.user_visible = user_visible;
            message.metadata.agent_visible = agent_visible;
        }
    }
    Ok(())
}

#[async_trait]
impl EffectHandler<GooseSession, OpenHumanEffect> for CheckpointRuntime {
    async fn apply_effects(
        &self,
        session: &GooseSession,
        effects: &mut [OpenHumanEffect],
        _emit: &goose_agent::operation::Emitter,
    ) -> Result<()> {
        let mut next = session.checkpoint.clone();
        for effect in effects.iter() {
            match effect {
                OpenHumanEffect::Conversation(effect) => match effect {
                    ConversationEffect::AppendMessage(message) => apply_conversation_effect(
                        &mut next,
                        ConversationEffect::AppendMessage(message.clone()),
                    )?,
                    ConversationEffect::ReplaceConversation(conversation) => {
                        apply_conversation_effect(
                            &mut next,
                            ConversationEffect::ReplaceConversation(conversation.clone()),
                        )?
                    }
                    ConversationEffect::PatchToolRequestMeta {
                        tool_call_id,
                        patch,
                    } => apply_conversation_effect(
                        &mut next,
                        ConversationEffect::PatchToolRequestMeta {
                            tool_call_id: tool_call_id.clone(),
                            patch: patch.clone(),
                        },
                    )?,
                    ConversationEffect::SetMessageVisibility {
                        message_id,
                        user_visible,
                        agent_visible,
                    } => apply_conversation_effect(
                        &mut next,
                        ConversationEffect::SetMessageVisibility {
                            message_id: message_id.clone(),
                            user_visible: *user_visible,
                            agent_visible: *agent_visible,
                        },
                    )?,
                },
                OpenHumanEffect::Usage {
                    model: _,
                    input_tokens,
                    output_tokens,
                    cached_input_tokens,
                    cache_creation_tokens,
                    reasoning_tokens,
                } => {
                    next.usage.primary_calls = next.usage.primary_calls.saturating_add(1);
                    next.usage.latest_primary_input_tokens = *input_tokens;
                    next.usage.cumulative_input_tokens = next
                        .usage
                        .cumulative_input_tokens
                        .saturating_add(*input_tokens);
                    next.usage.cumulative_output_tokens = next
                        .usage
                        .cumulative_output_tokens
                        .saturating_add(*output_tokens);
                    next.usage.cumulative_cached_input_tokens = next
                        .usage
                        .cumulative_cached_input_tokens
                        .saturating_add(*cached_input_tokens);
                    next.usage.cumulative_cache_creation_tokens = next
                        .usage
                        .cumulative_cache_creation_tokens
                        .saturating_add(*cache_creation_tokens);
                    next.usage.cumulative_reasoning_tokens = next
                        .usage
                        .cumulative_reasoning_tokens
                        .saturating_add(*reasoning_tokens);
                }
            }
        }
        let expected = session.checkpoint.revision;
        next.revision = expected.saturating_add(1);
        self.store
            .compare_and_swap(&session.id, expected, next)
            .await
            .context("persist Goose state-machine step")
    }
}
