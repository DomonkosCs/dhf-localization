class SimulationData:
    def __init__(self, x_odom, x_true, x_amcl, measurement):
        self.x_odom = x_odom
        self.x_true = x_true
        self.x_amcl = x_amcl
        self.measurement = measurement

        self.simulation_steps = len(self.x_odom)
