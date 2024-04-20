import flwr as fl

# Define strategy
class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()
        self.dropout_rate = 0.5  # Initialize dropout rate

    def on_fit_config(self, rnd: int) -> dict:
        """Send configuration to client. The dropout rate is adjusted dynamically based on random_var."""
        return {"dropout_rate": self.dropout_rate}

    def aggregate_evaluate(self, rnd, results, failures):
        # Call the parent class's aggregation method
        aggregated = super().aggregate_evaluate(rnd, results, failures)

        if aggregated is not None:
            loss, accuracy = aggregated
            print(f"Round {rnd}, Loss: {loss}, Accuracy: {accuracy}")

        # Extract random_vars and calculate their average
        random_vars = []
        print(f"Random Variables from Round {rnd}:")
        for client_id, result in results:
            if 'random_var' in result.metrics:
                random_var = result.metrics['random_var']
                random_vars.append(random_var)
                print(f"Client {client_id}: {random_var}")

        if random_vars:
            avg_random_var = sum(random_vars) / len(random_vars)
            print(f"Average Random Var from Round {rnd}: {avg_random_var}")

            # Adjust dropout rate based on the average of random_vars
            if avg_random_var > 0:
                self.dropout_rate = min(self.dropout_rate + 0.05, 0.9)
            else:
                self.dropout_rate = max(self.dropout_rate - 0.05, 0.1)
            print(f"New Dropout Rate for Round {rnd+1}: {self.dropout_rate}")

        return aggregated

# Start Flower server with enhanced logging
strategy = CustomStrategy()
server_config = fl.server.ServerConfig(num_rounds=100)
fl.server.start_server(server_address="0.0.0.0:8080", config=server_config, strategy=strategy)
