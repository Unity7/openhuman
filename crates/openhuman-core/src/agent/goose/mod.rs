//! OpenHuman host adapter for the pinned Goose Agent state machine.
//!
//! This module is deliberately a boundary, not another agent loop. Goose owns
//! step selection and re-entry. OpenHuman owns provider invocation, tools,
//! authorization, progress, cancellation and durable checkpoints.

mod convert;
mod inference;
mod runner;
mod store;
mod tools;
mod types;

pub use runner::GooseTurnAdapter;
pub use store::{GooseCheckpointStore, InMemoryGooseCheckpointStore};
pub use tools::GooseToolSecurity;
pub use types::{
    AcceptedToolAction, GooseCheckpoint, GooseStopReason, GooseTurnOutcome, GooseUsage,
    ToolObservation,
};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
