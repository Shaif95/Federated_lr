import flwr as fl

# Define strategy
class SaveMetricsStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self, rnd, results, failures):
        aggregated = super().aggregate_evaluate(rnd, results, failures)
        if aggregated is not None:
            loss, accuracy = aggregated
            print(f"Round {rnd}, Loss: {loss}, Accuracy: {accuracy}")
        return aggregated

# Start Flower server with enhanced logging
strategy = SaveMetricsStrategy()
server_config = fl.server.ServerConfig(num_rounds=10)  # Create ServerConfig object

fl.server.start_server(
    server_address="0.0.0.0:8080", 
    config=server_config,  # Pass ServerConfig object instead of dictionary
    strategy=strategy
)
