import json
import os.path


class ScrappingConfig:
    def __init__(self):
        self.start_lat = 30
        self.end_lat = 30
        self.start_lon = 120
        self.end_lon = 120
        self.fov = 36
        self.dis = 800
        self.alt = 0
        self.lat_step = 0.01
        self.lon_step = 0.01

    @property
    def num_lat(self):
        return int((self.end_lat - self.start_lat) / self.lat_step) + 1

    @property
    def num_lon(self):
        return int((self.end_lon - self.start_lon) / self.lon_step) + 1

    def save_to_file(self, file_path):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @staticmethod
    def load_from_file(config_path):
        config = ScrappingConfig()
        with open(config_path, 'r') as f:
            c = json.load(f)

        config.start_lat, config.start_lon = c['start_lat'], c['start_lon']
        config.end_lat, config.end_lon = c['end_lat'], c['end_lon']
        config.fov, config.dis, config.alt = c['fov'], c['dis'], c['alt']
        config.lat_step, config.lon_step = c["lat_step"], c["lon_step"]
        return config
