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

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config={"num_rounds": 100},  # Use dictionary for configuration in Flower 0.19.0
    strategy=strategy
)
