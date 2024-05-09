from smilesrnn.RL import (
    AugmentedHillClimb,
    BestAgentReminder,
    HillClimb,
    HillClimbRegularized,
    Reinforce,
    ReinforceRegularized,
    Reinvent,
)

RL_strategies = [
    Reinforce,
    ReinforceRegularized,
    Reinvent,
    BestAgentReminder,
    HillClimb,
    HillClimbRegularized,
    AugmentedHillClimb,
]
