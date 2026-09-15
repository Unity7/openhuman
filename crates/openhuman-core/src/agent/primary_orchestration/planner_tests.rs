use std::collections::HashSet;

use super::*;
use crate::agent::primary_orchestration::capability::{
    validate_routes, CapabilityAvailability, CapabilityBackend, CapabilityModality,
    CapabilityOperation, CapabilityPlan, CapabilityPolicy, CapabilitySideEffect,
    CapabilityValidationError, MonetaryBoundary, ToolCapability, ToolRoute,
};
use crate::agent::primary_orchestration::catalog::CapabilityCatalog;
use crate::agent::primary_orchestration::intent::{
    IntentCompletion, PrimaryIntentFamily, RequestIntent,
};
use crate::agent::primary_orchestration::mode::PrimaryTurnMode;
use crate::tools::traits::PermissionLevel;

fn make_test_capability(
    name: &str,
    operations: Vec<CapabilityOperation>,
    modalities: Vec<CapabilityModality>,
    backend: CapabilityBackend,
    monetary_boundary: MonetaryBoundary,
    availability: CapabilityAvailability,
    permission: PermissionLevel,
    priority: u16,
    registration_index: usize,
) -> ToolCapability {
    ToolCapability {
        name: name.to_string(),
        operations,
        modalities,
        backend,
        monetary_boundary,
        side_effect: CapabilitySideEffect::None,
        availability,
        permission,
        priority,
        registration_index,
    }
}

fn make_test_route(
    name: &str,
    operations: Vec<CapabilityOperation>,
    modalities: Vec<CapabilityModality>,
    backend: CapabilityBackend,
    monetary_boundary: MonetaryBoundary,
    availability: CapabilityAvailability,
    permission: PermissionLevel,
    priority: u16,
    registration_index: usize,
) -> ToolRoute {
    ToolRoute::new(make_test_capability(
        name,
        operations,
        modalities,
        backend,
        monetary_boundary,
        availability,
        permission,
        priority,
        registration_index,
    ))
}

fn permissive_policy() -> CapabilityPolicy {
    CapabilityPolicy::new(PermissionLevel::ReadOnly, true, true, true)
}

fn make_catalog(routes: Vec<ToolRoute>) -> CapabilityCatalog {
    CapabilityCatalog {
        routes,
        diagnostics: Vec::new(),
    }
}

fn make_intent(
    family: PrimaryIntentFamily,
    operations: Vec<CapabilityOperation>,
    modalities: Vec<CapabilityModality>,
) -> RequestIntent {
    RequestIntent {
        family,
        operations,
        modalities,
        completion: IntentCompletion::FinalText,
        known_url: None,
        explicit_memory: family == PrimaryIntentFamily::Memory,
        explicit_generation: family == PrimaryIntentFamily::ImageGeneration,
        explicit_retrieval: family == PrimaryIntentFamily::ImageRetrieval,
    }
}

fn combine_downward_override(
    persisted: CapabilityPolicy,
    override_policy: CapabilityPolicy,
) -> CapabilityPolicy {
    CapabilityPolicy::new(
        persisted.max_permission.min(override_policy.max_permission),
        persisted.allow_managed_metered && override_policy.allow_managed_metered,
        persisted.allow_byok && override_policy.allow_byok,
        persisted.allow_external_network && override_policy.allow_external_network,
    )
}

#[test]
fn test_chat_mode_always_plans_zero_tools() {
    let routes = vec![
        make_test_route(
            "web_search",
            vec![CapabilityOperation::SearchWeb],
            vec![CapabilityModality::Text],
            CapabilityBackend::DirectNetwork,
            MonetaryBoundary::NonMetered,
            CapabilityAvailability::Available,
            PermissionLevel::ReadOnly,
            0,
            0,
        ),
        make_test_route(
            "file_read",
            vec![CapabilityOperation::ReadWorkspace],
            vec![CapabilityModality::File],
            CapabilityBackend::Local,
            MonetaryBoundary::NonMetered,
            CapabilityAvailability::Available,
            PermissionLevel::ReadOnly,
            0,
            1,
        ),
        make_test_route(
            "generate_image",
            vec![CapabilityOperation::GenerateImage],
            vec![CapabilityModality::Image],
            CapabilityBackend::Managed,
            MonetaryBoundary::ManagedMetered,
            CapabilityAvailability::Available,
            PermissionLevel::ReadOnly,
            0,
            2,
        ),
        make_test_route(
            "memory_recall",
            vec![CapabilityOperation::RecallMemory],
            vec![CapabilityModality::Memory],
            CapabilityBackend::Local,
            MonetaryBoundary::NonMetered,
            CapabilityAvailability::Available,
            PermissionLevel::ReadOnly,
            0,
            3,
        ),
    ];
    let catalog = make_catalog(routes);
    let policy = permissive_policy();

    let families = [
        (PrimaryIntentFamily::Conversation, vec![], vec![]),
        (
            PrimaryIntentFamily::Web,
            vec![CapabilityOperation::SearchWeb],
            vec![CapabilityModality::Text],
        ),
        (
            PrimaryIntentFamily::Repository,
            vec![CapabilityOperation::ReadWorkspace],
            vec![CapabilityModality::File],
        ),
        (
            PrimaryIntentFamily::ImageGeneration,
            vec![CapabilityOperation::GenerateImage],
            vec![CapabilityModality::Image],
        ),
        (
            PrimaryIntentFamily::Memory,
            vec![CapabilityOperation::RecallMemory],
            vec![CapabilityModality::Memory],
        ),
    ];

    for (family, ops, mods) in families {
        let intent = make_intent(family, ops, mods);
        let plan = plan_capabilities(CapabilityPlannerInput::new(
            PrimaryTurnMode::Chat,
            &intent,
            &catalog,
            None,
            policy,
        ))
        .expect("chat planning should succeed");

        assert!(
            plan.routes.is_empty(),
            "chat mode must always plan zero tools for family {family:?}"
        );
        assert!(
            plan.unavailable_reason.is_none(),
            "chat mode must not emit unavailable reason"
        );
        assert!(
            plan.enabled_names().is_empty(),
            "chat mode enabled names must be empty"
        );
    }
}

#[test]
fn test_deterministic_route_ordering() {
    let managed = make_test_route(
        "managed_search",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::Managed,
        MonetaryBoundary::ManagedMetered,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        0,
        0,
    );
    let byok = make_test_route(
        "byok_search",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::Byok,
        MonetaryBoundary::UserSuppliedKey,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        0,
        1,
    );
    let network = make_test_route(
        "network_search",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::DirectNetwork,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        0,
        2,
    );
    let browser = make_test_route(
        "browser_search",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::LocalBrowser,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        0,
        3,
    );
    let local = make_test_route(
        "local_search",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::Local,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        0,
        4,
    );

    // Provide in reverse order of expected rank
    let catalog = make_catalog(vec![managed, byok, network, browser, local]);
    let policy = permissive_policy();
    let intent = make_intent(
        PrimaryIntentFamily::Web,
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
    );

    let plan = plan_capabilities(CapabilityPlannerInput::new(
        PrimaryTurnMode::Agent,
        &intent,
        &catalog,
        None,
        policy,
    ))
    .expect("planning should succeed");

    let ordered_names: Vec<&str> = plan
        .routes
        .iter()
        .map(|r| r.capability.name.as_str())
        .collect();

    assert_eq!(
        ordered_names,
        vec![
            "local_search",
            "browser_search",
            "network_search",
            "byok_search",
            "managed_search",
        ],
        "routes must sort Local -> LocalBrowser -> DirectNetwork -> Byok -> Managed"
    );

    // Test priority and registration index sub-ordering within same backend
    let local_high_priority = make_test_route(
        "local_p10",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::Local,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        10,
        1,
    );
    let local_low_priority = make_test_route(
        "local_p20",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::Local,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        20,
        0,
    );
    let local_same_priority_earlier_reg = make_test_route(
        "local_p10_idx0",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::Local,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        10,
        0,
    );

    let catalog_sub = make_catalog(vec![
        local_low_priority,
        local_high_priority,
        local_same_priority_earlier_reg,
    ]);

    let plan_sub = plan_capabilities(CapabilityPlannerInput::new(
        PrimaryTurnMode::Agent,
        &intent,
        &catalog_sub,
        None,
        policy,
    ))
    .expect("planning should succeed");

    let sub_names: Vec<&str> = plan_sub
        .routes
        .iter()
        .map(|r| r.capability.name.as_str())
        .collect();

    assert_eq!(
        sub_names,
        vec!["local_p10_idx0", "local_p10", "local_p20"],
        "within same backend, sort key must be (priority, registration_index)"
    );
}

#[test]
fn test_managed_metered_policy_and_downward_override() {
    let managed_route = make_test_route(
        "managed_search",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::Managed,
        MonetaryBoundary::ManagedMetered,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        0,
        0,
    );
    let catalog = make_catalog(vec![managed_route.clone()]);
    let intent = make_intent(
        PrimaryIntentFamily::Web,
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
    );

    // Direct policy check on capability
    let policy_managed_true = CapabilityPolicy::new(PermissionLevel::ReadOnly, true, true, true);
    let policy_managed_false = CapabilityPolicy::new(PermissionLevel::ReadOnly, false, true, true);
    assert!(policy_managed_true.allows(&managed_route.capability));
    assert!(!policy_managed_false.allows(&managed_route.capability));

    // Downward override truth table:
    // (persisted, downward_override) -> effective
    // (false, true) -> false (override cannot broaden false persisted)
    // (false, false) -> false
    // (true, false) -> false (override can narrow true persisted)
    // (true, true) -> true
    let cases = [
        (false, true, false),
        (false, false, false),
        (true, false, false),
        (true, true, true),
    ];

    for (persisted_allow, override_allow, expected_effective) in cases {
        let persisted_policy =
            CapabilityPolicy::new(PermissionLevel::ReadOnly, persisted_allow, true, true);
        let override_policy =
            CapabilityPolicy::new(PermissionLevel::ReadOnly, override_allow, true, true);
        let effective_policy = combine_downward_override(persisted_policy, override_policy);

        assert_eq!(
            effective_policy.allow_managed_metered, expected_effective,
            "persisted={persisted_allow}, override={override_allow} must yield effective={expected_effective}"
        );

        let plan = plan_capabilities(CapabilityPlannerInput::new(
            PrimaryTurnMode::Agent,
            &intent,
            &catalog,
            None,
            effective_policy,
        ))
        .expect("planning should evaluate");

        if expected_effective {
            assert_eq!(
                plan.routes.len(),
                1,
                "managed route must be planned when effective allow_managed_metered is true"
            );
            assert_eq!(plan.routes[0].capability.name, "managed_search");
            assert!(plan.unavailable_reason.is_none());
        } else {
            assert!(
                plan.routes.is_empty(),
                "managed route must be omitted when effective allow_managed_metered is false"
            );
            assert_eq!(
                plan.unavailable_reason,
                Some("capability unavailable for web".to_string()),
                "missing route must produce deterministic unavailable reason"
            );
        }
    }
}

#[test]
fn test_omission_of_unavailable_unhealthy_disallowed_and_ceiling_excluded() {
    let disabled_route = make_test_route(
        "disabled_tool",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::Local,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Disabled,
        PermissionLevel::ReadOnly,
        0,
        0,
    );
    let unconfigured_route = make_test_route(
        "unconfigured_tool",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::Local,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Unconfigured,
        PermissionLevel::ReadOnly,
        0,
        1,
    );
    let unhealthy_route = make_test_route(
        "unhealthy_tool",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::Local,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Unhealthy,
        PermissionLevel::ReadOnly,
        0,
        2,
    );
    let permission_disallowed = make_test_route(
        "high_permission_tool",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::Local,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        0,
        3,
    );
    let byok_disallowed = make_test_route(
        "byok_disallowed_tool",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::Byok,
        MonetaryBoundary::UserSuppliedKey,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        0,
        4,
    );
    let network_disallowed = make_test_route(
        "network_disallowed_tool",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::DirectNetwork,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        0,
        5,
    );
    let ceiling_excluded = make_test_route(
        "ceiling_excluded_tool",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::Local,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        0,
        6,
    );
    let valid_retained = make_test_route(
        "valid_retained_tool",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::Local,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        0,
        7,
    );

    let catalog = make_catalog(vec![
        disabled_route,
        unconfigured_route,
        unhealthy_route,
        permission_disallowed,
        byok_disallowed,
        network_disallowed,
        ceiling_excluded,
        valid_retained,
    ]);

    let intent = make_intent(
        PrimaryIntentFamily::Web,
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
    );

    // Policy permits ReadOnly, disallows BYOK and external network
    let policy = CapabilityPolicy::new(PermissionLevel::ReadOnly, true, false, false);
    let ceiling = HashSet::from(["valid_retained_tool".to_string()]);

    let plan = plan_capabilities(CapabilityPlannerInput::new(
        PrimaryTurnMode::Agent,
        &intent,
        &catalog,
        Some(&ceiling),
        policy,
    ))
    .expect("planning should succeed");

    assert_eq!(
        plan.routes.len(),
        1,
        "only the valid, healthy, policy-allowed, ceiling-permitted route must be retained"
    );
    assert_eq!(plan.routes[0].capability.name, "valid_retained_tool");
    assert!(plan.unavailable_reason.is_none());

    // Also verify permission gate specifically: max_permission = None excludes ReadOnly
    let strict_permission_policy = CapabilityPolicy::new(PermissionLevel::None, true, true, true);
    let ceiling_all = HashSet::from(["valid_retained_tool".to_string()]);
    let plan_strict = plan_capabilities(CapabilityPlannerInput::new(
        PrimaryTurnMode::Agent,
        &intent,
        &catalog,
        Some(&ceiling_all),
        strict_permission_policy,
    ))
    .expect("planning should succeed");

    assert!(
        plan_strict.routes.is_empty(),
        "permission exceeding max_permission must be omitted"
    );
    assert_eq!(
        plan_strict.unavailable_reason,
        Some("capability unavailable for web".to_string())
    );
}

#[test]
fn test_image_retrieval_and_generation_separation() {
    let retrieval_route = make_test_route(
        "brave_image_search",
        vec![CapabilityOperation::RetrieveImage],
        vec![CapabilityModality::Image],
        CapabilityBackend::Byok,
        MonetaryBoundary::UserSuppliedKey,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        0,
        0,
    );
    let generation_route = make_test_route(
        "media_generate_image",
        vec![CapabilityOperation::GenerateImage],
        vec![CapabilityModality::Image],
        CapabilityBackend::Managed,
        MonetaryBoundary::ManagedMetered,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        0,
        1,
    );

    let catalog_both = make_catalog(vec![retrieval_route.clone(), generation_route.clone()]);
    let catalog_gen_only = make_catalog(vec![generation_route.clone()]);
    let catalog_retrieval_only = make_catalog(vec![retrieval_route.clone()]);
    let policy = permissive_policy();

    // Retrieval intent
    let retrieval_intent = make_intent(
        PrimaryIntentFamily::ImageRetrieval,
        vec![
            CapabilityOperation::SearchWeb,
            CapabilityOperation::FetchUrl,
            CapabilityOperation::RetrieveImage,
        ],
        vec![CapabilityModality::Image, CapabilityModality::WebPage],
    );

    // Retrieval with both tools in catalog selects ONLY retrieval
    let plan_retrieval = plan_capabilities(CapabilityPlannerInput::new(
        PrimaryTurnMode::Agent,
        &retrieval_intent,
        &catalog_both,
        None,
        policy,
    ))
    .expect("planning should succeed");
    assert_eq!(plan_retrieval.routes.len(), 1);
    assert_eq!(
        plan_retrieval.routes[0].capability.name,
        "brave_image_search"
    );
    assert!(plan_retrieval.unavailable_reason.is_none());

    // Retrieval with ONLY generation tool returns zero routes and unavailable reason
    let plan_retrieval_missing = plan_capabilities(CapabilityPlannerInput::new(
        PrimaryTurnMode::Agent,
        &retrieval_intent,
        &catalog_gen_only,
        None,
        policy,
    ))
    .expect("planning should succeed");
    assert!(
        plan_retrieval_missing.routes.is_empty(),
        "image retrieval must never select image generation"
    );
    assert_eq!(
        plan_retrieval_missing.unavailable_reason,
        Some("capability unavailable for image_retrieval".to_string())
    );

    // Generation intent
    let generation_intent = make_intent(
        PrimaryIntentFamily::ImageGeneration,
        vec![CapabilityOperation::GenerateImage],
        vec![CapabilityModality::Image],
    );

    // Generation with both tools in catalog selects ONLY generation
    let plan_generation = plan_capabilities(CapabilityPlannerInput::new(
        PrimaryTurnMode::Agent,
        &generation_intent,
        &catalog_both,
        None,
        policy,
    ))
    .expect("planning should succeed");
    assert_eq!(plan_generation.routes.len(), 1);
    assert_eq!(
        plan_generation.routes[0].capability.name,
        "media_generate_image"
    );
    assert!(plan_generation.unavailable_reason.is_none());

    // Generation with ONLY retrieval tool returns zero routes and unavailable reason
    let plan_generation_missing = plan_capabilities(CapabilityPlannerInput::new(
        PrimaryTurnMode::Agent,
        &generation_intent,
        &catalog_retrieval_only,
        None,
        policy,
    ))
    .expect("planning should succeed");
    assert!(
        plan_generation_missing.routes.is_empty(),
        "image generation must never select image retrieval"
    );
    assert_eq!(
        plan_generation_missing.unavailable_reason,
        Some("capability unavailable for image_generation".to_string())
    );
}

#[test]
fn test_memory_selects_only_ready_memory_route() {
    let memory_ready = make_test_route(
        "memory_recall",
        vec![CapabilityOperation::RecallMemory],
        vec![CapabilityModality::Memory],
        CapabilityBackend::Local,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        0,
        0,
    );
    let memory_unhealthy = make_test_route(
        "memory_store",
        vec![CapabilityOperation::StoreMemory],
        vec![CapabilityModality::Memory],
        CapabilityBackend::Local,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Unhealthy,
        PermissionLevel::ReadOnly,
        0,
        1,
    );
    let memory_disabled = make_test_route(
        "memory_forget",
        vec![CapabilityOperation::StoreMemory],
        vec![CapabilityModality::Memory],
        CapabilityBackend::Local,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Disabled,
        PermissionLevel::ReadOnly,
        0,
        2,
    );
    let web_tool = make_test_route(
        "web_search",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::DirectNetwork,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        0,
        3,
    );
    let file_tool = make_test_route(
        "file_read",
        vec![CapabilityOperation::ReadWorkspace],
        vec![CapabilityModality::File],
        CapabilityBackend::Local,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        0,
        4,
    );

    let catalog = make_catalog(vec![
        memory_ready.clone(),
        memory_unhealthy,
        memory_disabled,
        web_tool,
        file_tool,
    ]);
    let policy = permissive_policy();

    // Recall intent
    let recall_intent = make_intent(
        PrimaryIntentFamily::Memory,
        vec![CapabilityOperation::RecallMemory],
        vec![CapabilityModality::Memory, CapabilityModality::Text],
    );
    let plan_recall = plan_capabilities(CapabilityPlannerInput::new(
        PrimaryTurnMode::Agent,
        &recall_intent,
        &catalog,
        None,
        policy,
    ))
    .expect("planning should succeed");

    assert_eq!(
        plan_recall.routes.len(),
        1,
        "memory intent must only select ready memory routes"
    );
    assert_eq!(plan_recall.routes[0].capability.name, "memory_recall");
    assert!(plan_recall.unavailable_reason.is_none());

    // Store intent where only unhealthy/disabled memory routes exist
    let store_intent = make_intent(
        PrimaryIntentFamily::Memory,
        vec![CapabilityOperation::StoreMemory],
        vec![CapabilityModality::Memory, CapabilityModality::Text],
    );
    let plan_store = plan_capabilities(CapabilityPlannerInput::new(
        PrimaryTurnMode::Agent,
        &store_intent,
        &catalog,
        None,
        policy,
    ))
    .expect("planning should succeed");

    assert!(
        plan_store.routes.is_empty(),
        "non-ready memory routes must be omitted"
    );
    assert_eq!(
        plan_store.unavailable_reason,
        Some("capability unavailable for memory".to_string()),
        "unready memory route must yield deterministic unavailable reason"
    );
}

#[test]
fn test_duplicate_route_names_are_rejected() {
    let route1 = make_test_route(
        "duplicate_name",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::Local,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        0,
        0,
    );
    let route2 = make_test_route(
        "duplicate_name",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::Local,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        0,
        1,
    );

    // Direct plan validation rejection
    let plan_err = CapabilityPlan::new(vec![route1.clone(), route2.clone()], None)
        .expect_err("plan with duplicate route names must fail validation");
    assert_eq!(
        plan_err,
        CapabilityValidationError::DuplicateName("duplicate_name".to_string())
    );

    // validate_routes helper rejection
    let routes_err = validate_routes(&[route1.clone(), route2.clone()])
        .expect_err("validate_routes with duplicates must fail");
    assert_eq!(
        routes_err,
        CapabilityValidationError::DuplicateName("duplicate_name".to_string())
    );

    // Planner rejection
    let catalog = make_catalog(vec![route1, route2]);
    let policy = permissive_policy();
    let intent = make_intent(
        PrimaryIntentFamily::Web,
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
    );

    let result = plan_capabilities(CapabilityPlannerInput::new(
        PrimaryTurnMode::Agent,
        &intent,
        &catalog,
        None,
        policy,
    ));

    assert_eq!(
        result,
        Err(CapabilityValidationError::DuplicateName(
            "duplicate_name".to_string()
        )),
        "planner must return DuplicateName validation error when duplicates are retained"
    );
}

#[test]
fn test_missing_same_boundary_route_unavailable_reason_and_no_cross_boundary() {
    // Catalog with ONLY repository routes
    let repo_route = make_test_route(
        "file_read",
        vec![CapabilityOperation::ReadWorkspace],
        vec![CapabilityModality::File],
        CapabilityBackend::Local,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        0,
        0,
    );
    let catalog = make_catalog(vec![repo_route]);
    let policy = permissive_policy();

    // Verify all other intent families return deterministic unavailable reason and never select repository tool
    let boundary_cases = [
        (
            PrimaryIntentFamily::Web,
            vec![CapabilityOperation::SearchWeb],
            vec![CapabilityModality::Text],
            "capability unavailable for web",
        ),
        (
            PrimaryIntentFamily::ImageRetrieval,
            vec![CapabilityOperation::RetrieveImage],
            vec![CapabilityModality::Image],
            "capability unavailable for image_retrieval",
        ),
        (
            PrimaryIntentFamily::ImageGeneration,
            vec![CapabilityOperation::GenerateImage],
            vec![CapabilityModality::Image],
            "capability unavailable for image_generation",
        ),
        (
            PrimaryIntentFamily::Memory,
            vec![CapabilityOperation::RecallMemory],
            vec![CapabilityModality::Memory],
            "capability unavailable for memory",
        ),
        (
            PrimaryIntentFamily::Scheduling,
            vec![CapabilityOperation::Schedule],
            vec![CapabilityModality::Schedule],
            "capability unavailable for scheduling",
        ),
        (
            PrimaryIntentFamily::Delegation,
            vec![CapabilityOperation::Delegate],
            vec![CapabilityModality::Text],
            "capability unavailable for delegation",
        ),
    ];

    for (family, ops, mods, expected_reason) in boundary_cases {
        let intent = make_intent(family, ops, mods);
        let plan = plan_capabilities(CapabilityPlannerInput::new(
            PrimaryTurnMode::Agent,
            &intent,
            &catalog,
            None,
            policy,
        ))
        .expect("planning should succeed");

        assert!(
            plan.routes.is_empty(),
            "missing {family:?} route must never cross boundary to select repository route"
        );
        assert_eq!(
            plan.unavailable_reason,
            Some(expected_reason.to_string()),
            "must return deterministic unavailable reason for {family:?}"
        );
    }

    // Now test missing repository route when catalog only has web
    let web_route = make_test_route(
        "web_search",
        vec![CapabilityOperation::SearchWeb],
        vec![CapabilityModality::Text],
        CapabilityBackend::DirectNetwork,
        MonetaryBoundary::NonMetered,
        CapabilityAvailability::Available,
        PermissionLevel::ReadOnly,
        0,
        0,
    );
    let catalog_web = make_catalog(vec![web_route]);
    let repo_intent = make_intent(
        PrimaryIntentFamily::Repository,
        vec![CapabilityOperation::ReadWorkspace],
        vec![CapabilityModality::File],
    );

    let plan_repo = plan_capabilities(CapabilityPlannerInput::new(
        PrimaryTurnMode::Agent,
        &repo_intent,
        &catalog_web,
        None,
        policy,
    ))
    .expect("planning should succeed");

    assert!(
        plan_repo.routes.is_empty(),
        "missing repository route must never cross boundary to select web route"
    );
    assert_eq!(
        plan_repo.unavailable_reason,
        Some("capability unavailable for repository".to_string())
    );

    // Test mode boundaries in Assist mode: Assist mode does not permit Repository, Scheduling, or Delegation
    let assist_disallowed_families = [
        (
            PrimaryIntentFamily::Repository,
            vec![CapabilityOperation::ReadWorkspace],
            vec![CapabilityModality::File],
            "capability unavailable for repository",
        ),
        (
            PrimaryIntentFamily::Scheduling,
            vec![CapabilityOperation::Schedule],
            vec![CapabilityModality::Schedule],
            "capability unavailable for scheduling",
        ),
        (
            PrimaryIntentFamily::Delegation,
            vec![CapabilityOperation::Delegate],
            vec![CapabilityModality::Text],
            "capability unavailable for delegation",
        ),
    ];

    let full_catalog = make_catalog(vec![
        make_test_route(
            "file_read",
            vec![CapabilityOperation::ReadWorkspace],
            vec![CapabilityModality::File],
            CapabilityBackend::Local,
            MonetaryBoundary::NonMetered,
            CapabilityAvailability::Available,
            PermissionLevel::ReadOnly,
            0,
            0,
        ),
        make_test_route(
            "schedule_tool",
            vec![CapabilityOperation::Schedule],
            vec![CapabilityModality::Schedule],
            CapabilityBackend::Local,
            MonetaryBoundary::NonMetered,
            CapabilityAvailability::Available,
            PermissionLevel::ReadOnly,
            0,
            1,
        ),
        make_test_route(
            "delegate_tool",
            vec![CapabilityOperation::Delegate],
            vec![CapabilityModality::Text],
            CapabilityBackend::Local,
            MonetaryBoundary::NonMetered,
            CapabilityAvailability::Available,
            PermissionLevel::ReadOnly,
            0,
            2,
        ),
    ]);

    for (family, ops, mods, expected_reason) in assist_disallowed_families {
        let intent = make_intent(family, ops, mods);
        let plan_assist = plan_capabilities(CapabilityPlannerInput::new(
            PrimaryTurnMode::Assist,
            &intent,
            &full_catalog,
            None,
            policy,
        ))
        .expect("planning should succeed");

        assert!(
            plan_assist.routes.is_empty(),
            "Assist mode must never permit {family:?} routes even when present in catalog"
        );
        assert_eq!(
            plan_assist.unavailable_reason,
            Some(expected_reason.to_string()),
            "Assist mode must return deterministic unavailable reason for {family:?}"
        );
    }
}
