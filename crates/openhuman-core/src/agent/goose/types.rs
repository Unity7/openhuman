use std::collections::BTreeMap;

use goose_provider_types::conversation::Conversation;
use serde::{Deserialize, Serialize};

/// Durable per-turn usage. `latest_primary_input_tokens` is context occupancy
/// for the latest primary call; the other fields are cumulative traffic.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GooseUsage {
    pub primary_calls: u32,
    pub latest_primary_input_tokens: u64,
    pub cumulative_input_tokens: u64,
    pub cumulative_output_tokens: u64,
    pub cumulative_cached_input_tokens: u64,
    pub cumulative_cache_creation_tokens: u64,
    pub cumulative_reasoning_tokens: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AcceptedToolAction {
    pub call_id: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
    pub observation: Option<ToolObservation>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolObservation {
    pub call_id: String,
    pub success: bool,
    pub output: String,
}

/// The complete state reloaded by Goose before every pass.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GooseCheckpoint {
    pub revision: u64,
    pub conversation: Conversation,
    #[serde(default)]
    pub actions: BTreeMap<String, AcceptedToolAction>,
    #[serde(default)]
    pub usage: GooseUsage,
}

impl GooseCheckpoint {
    pub fn new(conversation: Conversation) -> Self {
        Self {
            revision: 0,
            conversation,
            actions: BTreeMap::new(),
            usage: GooseUsage::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GooseStopReason {
    FinalAnswer,
    Cancelled,
    Yielded,
}

#[derive(Clone, Debug)]
pub struct GooseTurnOutcome {
    pub checkpoint: GooseCheckpoint,
    pub stop_reason: GooseStopReason,
    pub openhuman_messages: Vec<crate::agent::messages::ConversationMessage>,
}

pub(super) enum OpenHumanEffect {
    Conversation(goose_agent::operation::ConversationEffect),
    Usage {
        model: String,
        input_tokens: u64,
        output_tokens: u64,
        cached_input_tokens: u64,
        cache_creation_tokens: u64,
        reasoning_tokens: u64,
    },
}

impl From<goose_provider_types::conversation::message::Message> for OpenHumanEffect {
    fn from(message: goose_provider_types::conversation::message::Message) -> Self {
        Self::Conversation(goose_agent::operation::ConversationEffect::AppendMessage(
            message,
        ))
    }
}

impl goose_agent::operation::MachineEffect for OpenHumanEffect {
    fn ensure_message_ids(&mut self) {
        if let Self::Conversation(effect) = self {
            effect.ensure_message_ids();
        }
    }
}

pub(super) struct GooseSession {
    pub id: String,
    pub checkpoint: GooseCheckpoint,
}

impl goose_agent::machine::MachineSession for GooseSession {
    fn id(&self) -> &str {
        &self.id
    }

    fn conversation(&self) -> Option<&Conversation> {
        Some(&self.checkpoint.conversation)
    }
}
