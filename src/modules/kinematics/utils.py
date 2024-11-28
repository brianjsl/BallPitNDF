from pydrake.all import (
    LeafSystem
)

class ExtractArmPositions(LeafSystem):
    def __init__(self):
        super().__init__()
        self.DeclareVectorInputPort("full_state_estimated", 18)
        self.DeclareVectorOutputPort("arm_state_estimated", 14, self.DoCalcOutput)
    
    def DoCalcOutput(self, context, output):
        full_state = self.get_input_port(0).Eval(context)
        arm_state = full_state[:14]
        output.SetFromVector(arm_state)