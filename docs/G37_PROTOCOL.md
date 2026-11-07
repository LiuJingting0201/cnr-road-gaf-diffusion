# G37: OD + class (+ bus stops) route-and-motion development pilot

Locked before route proposal outcomes, 2026-11-07. User explicitly expands scope beyond fixed routes. This finite pilot is not a new blind test or a learned routing model.

## Inputs and separation
Use first four requests by ID in each of five classes from G36 morning manifest (20 total), without filtering by map or reference success. A separate preparation step redacts each request to ID, label/vClass, network hash, start/end lane+position and ordered public bus stop lane+position. Interior itinerary edges, future FCD, target duration and reference scores are forbidden generator inputs. Bus endpoints plus ordered service stops is a distinct declared condition from other vehicles' OD; report diversity separately. No frozen test access. Existing G36 reference is development evidence only and contains one itinerary per OD, not a route-choice distribution.

## Routing pilot
Build a directed normal-edge graph from vehicle-permitted lanes and explicit connected junction chains (2m gap guard), using network-only length costs. Up to12 deterministic penalized shortest-path proposals per request, retain at most4 distinct successfully constructed lane-corridor routes. Existing G32 geometry constructor enforces endpoint lanes/positions, internal connections and ordered stop anchors. Keep routes within1.5x shortest supported geometry in the retained catalog; rejected alternatives and all denominators reported. Route proposal is algorithmic, not diffusion-learned choice. All arms share the same route catalog, hence proposal/geometry work is not hidden in one arm's budget.

Compare minimum-length supported catalog route ×8 motion draws versus up to4 catalog routes allocated round-robin across8 draws. Use current native diffusion seed42,100DDIM,8candidate slots per OD in both arms. Same candidate IDs pair duration-residual/noise keys across arms, but route-conditioned lengths change prior duration and sequence length: equal candidate/denoiser-call budget is not equal elapsed time or token count. Within each pool compare Uniform/S/V/SV from the same frozen current GAF readers. Equal-log dual, ties/shared failures follow inherited permutation. No teacher score chooses proposals or removes routes.

## Readouts and limits
Primary feasibility:20-request coverage, viable routes/class, edge-weighted pair diversity, detour ratio, endpoints and stop-anchor retention. GAF role: same-route vs multi-route image diversity, route-switch frequency, selected length/detour and S/V/SV disagreement; own target score auxiliary only. Independent physical summaries and native speed/acceleration W1 to existing SUMO reference report every class, denominators and all four selectors. Reference itinerary overlap is not a correctness label; alternative valid routes are not errors simply because they differ. No claim of realistic route-choice probabilities from one reference trip. No downstream utility/in-loop benefit claim.

On feasibility success, next experiment must learn route preferences from training-side route observations, add matched non-GAF routing feedback, freeze independent OD/episode evaluation, and distinguish public service conditions from general vehicle OD. Do not rewrite thesis as successful OD synthesis before results.

Cloud: isolated g37-job; up to2hours,8GiB output budget,12GiB minimum free; exclusive lock/complete/failed markers. No existing worktree mutation. Git stores code, protocol and small aggregate results only.

Technical references: https://sumo.dlr.de/docs/Simulation/Routing.html ; https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html ; https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html
