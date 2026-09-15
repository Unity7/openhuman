//! Deterministic primary interactive-turn routing for Phase 7.
//!
//! This module decides only *which* execution shape owns a turn.  It does not
//! select tools or infer capability metadata; that is the following gate.

pub mod capability;
mod mode;
mod runtime;

pub use capability::{
    is_backend_monetary_compatible, validate_routes, CapabilityAvailability, CapabilityBackend,
    CapabilityModality, CapabilityOperation, CapabilityPlan, CapabilityPolicy,
    CapabilitySideEffect, CapabilityValidationError, MonetaryBoundary, ToolCapability, ToolRoute,
};
pub use mode::{
    resolve_orchestration_engine, resolve_primary_turn_mode, ModeResolutionInput, PrimaryTurnMode,
    LOCAL_QWEN_PROVIDER_BINDING,
};
pub(crate) use runtime::{
    clear_primary_checkpoint, has_live_primary_checkpoint, primary_checkpoint_store,
};
