import torch
def convert_ez_to_nemo(self, restore_config):
        self.save_dir = f"{self.config.loggers.tb.save_dir}/temp_checkpoint"
        untar(restore_config.path, self.save_dir)
        self.model_config_path = self.save_dir + "/model_config.yaml"
        self.checkpoint_config = OmegaConf.load(self.model_config_path)
        self.model_weights_path = self.save_dir + "/model_weights.ckpt"
        weights = torch.load(self.model_weights_path)
        weight_dict = dict()

        print(self.checkpoint_config.keys())
        for i in restore_config.include:
            temp_dict = dict()
            weight_dict[i] = dict()
            for j in weights:
                if i == j.split(".")[0]:
                    weight_dict[i][".".join(j.split(".")[1:])] = weights[j]
        checkpoint = {
            "state_dict": weight_dict,
            
            "hyper_parameters": self.checkpoint_config        
            
        }
    