import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from datetime import datetime
import time
import joblib

class Modeling:
    def __init__(self,data):
        h2o.init()
        # Extract Data
        self.data = h2o.H2OFrame(data)
        self.X = data.columns[:-1]
        self.y = data.columns[-1]



    def save_model(self,model):
        model_path = h2o.save_model(model=model, path="project/models/trained_models/", force=True)
        print(f"Model saved at: {model_path}")




    def model_train(self):
        start = time.time()
        train, test = self.data.split_frame(ratios=[0.8], seed=7)
        aml = H2OAutoML(
            max_runtime_secs=600,
            seed=123,
            exclude_algos=["DeepLearning"],
            project_name="predictive_maintenance"
            )
        aml.train(x=self.X.tolist(), y=self.y, training_frame=train)
        # Saving Result In Logs
        leaderboard = aml.leaderboard.as_data_frame()
        leader_model = aml.leader
        # Saving The Model
        self.save_model(leader_model)


        test_performance = leader_model.model_performance(test)
        try:
         varimp = leader_model.varimp(use_pandas=True)
        except AttributeError:
            varimp = "Variable importance not available for this model."
        
        training_time = time.time()-start
        with open("project/models/logs/training_logs.txt", "a") as log_file:
            log_file.write(f"==== H2O AutoML Training Log {datetime.now()} ====\n")
            log_file.write("\n**Leaderboard:**\n")
            log_file.write(leaderboard.to_string(index=False))
            
            log_file.write("\n\n**Best Model Details:**\n")
            log_file.write(f"Model ID: {leader_model.model_id}\n")
            log_file.write(f"Model Type: {leader_model.algo}\n")
            
            log_file.write("\n**Test Set Performance:**\n")
            log_file.write(test_performance.__str__())
            
            log_file.write("\n\n**Variable Importance:**\n")
            log_file.write(str(varimp))
            log_file.write(f"Training Time: {training_time} s")
            log_file.write("\n\n==== End of Log ====")

        print("Training results saved to training_logs.txt")
        h2o.shutdown()

        
