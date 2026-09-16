//! OpenHuman host adapter for the pinned Goose Agent state machine.
//!
//! This module is deliberately a boundary, not another agent loop. Goose owns
//! step selection and re-entry. OpenHuman owns provider invocation, tools,
//! authorization, progress, cancellation and durable checkpoints.

mod convert;
mod inference;
pub mod qwen;
mod runner;
mod store;
mod tools;
mod types;

pub use qwen::{normalize_qwen_response, NormalizedQwenResponse, QwenInvalidCall};
pub use runner::GooseTurnAdapter;
pub use store::{FileGooseCheckpointStore, GooseCheckpointStore, InMemoryGooseCheckpointStore};
pub use tools::GooseToolSecurity;
pub use types::{
    AcceptedToolAction, GooseCheckpoint, GooseStopReason, GooseTurnOutcome, GooseUsage,
    ToolObservation,
};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(test)]
#[path = "qwen_tests.rs"]
#[rustfmt::skip]
mod qwen_tests;
