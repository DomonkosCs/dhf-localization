class SimulationData:
    def __init__(self, x_odom, x_true, x_amcl, measurement, times):
        self.x_odom = x_odom
        self.x_true = x_true
        self.x_amcl = x_amcl
        self.measurement = measurement
        self.times = times

        self.simulation_steps = len(self.x_odom)
