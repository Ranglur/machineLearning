import numpy as np
import matplotlib.pyplot as plt


## Helper functions:
def generate_data(w: float, b: float, sigma: float, interval: np.ndarray = np.array([0,1]), numDatapoints: int = 10) -> np.ndarray:
    # Generate uniform x-values within the given interval
    x_values = np.random.uniform(interval[0], interval[1], numDatapoints)
    
    # Generate the corresponding y-values using the linear relation with added Gaussian noise
    y_values = w * x_values + b + np.random.normal(0, sigma, numDatapoints)
    
    
    return x_values, y_values

def calculate_analytical_solution(x_values, y_values):
    """
    Calculates the weight (w) and bias (b) using NumPy's linear algebra functions.

    Parameters:
    -----------
    x_values : np.ndarray
        The input feature data.
    y_values : np.ndarray
        The target/output data.

    Returns:
    --------
    tuple
        A tuple containing the analytical weight (w) and bias (b).
    """
    # Add a column of ones to X to account for the bias term
    X = np.vstack([x_values, np.ones(len(x_values))]).T
    y = y_values

    # Calculate the analytical solution using the normal equation
    theta = np.linalg.inv(X.T @ X) @ X.T @ y

    # The first element of theta is w, the second is b
    w_analytical = theta[0]
    b_analytical = theta[1]

    return w_analytical, b_analytical


## ML model
class LinearRegressionMLModel:
    """
    A simple implementation of linear regression using manual gradient descent.

    Attributes:
    -----------
    x_values : np.ndarray
        The input feature data.
    y_values : np.ndarray
        The target/output data.
    w : float
        The weight (slope) of the linear model.
    b : float
        The bias (intercept) of the linear model.
    alpha : float
        The learning rate for gradient descent.
    N : int
        The number of data points in the dataset.

    Methods:
    --------
    forward():
        Performs a forward pass, calculating the predicted values based on current weights and bias.
    loss(y_prediction):
        Calculates the mean squared error (MSE) between the predicted and actual values.
    back(y_prediction):
        Performs backpropagation, updating the weights and bias based on the calculated gradients.
    train(epochs: int, showProgression=True):
        Trains the model for a specified number of epochs, optionally showing progress at regular intervals.
    __call__(x_input):
        Allows the model to be called like a function to make predictions.
    set_weight(w):
        Manually sets the weight of the model.
    set_bias(b):
        Manually sets the bias of the model.
    """

    def __init__(self, x_values=np.array([]), y_values=np.array([]), alpha=0.01) -> None:
        """
        Initializes the linear regression model with input data, target data, and learning rate.

        Parameters:
        -----------
        x_values : np.ndarray
            The input feature data.
        y_values : np.ndarray
            The target/output data.
        alpha : float
            The learning rate for gradient descent. Default is 0.01.
        """
        self.x_values = x_values
        self.y_values = y_values
        self.w = 0  # Initialize weight to 0
        self.b = 0  # Initialize bias to 0
        self.alpha = alpha  # Set the learning rate
        self.N = len(x_values)  # Number of data points

    def forward(self):
        """
        Performs a forward pass, calculating the predicted values.

        Returns:
        --------
        np.ndarray
            The predicted values based on current weight and bias.
        """
        return self.w * self.x_values + self.b

    def loss(self, y_prediction):
        """
        Calculates the mean squared error (MSE) between the predicted values and the actual values.

        Parameters:
        -----------
        y_prediction : np.ndarray
            The predicted values from the forward pass.

        Returns:
        --------
        float
            The mean squared error (MSE).
        """
        return np.mean((y_prediction - self.y_values) ** 2)

    def back(self, y_prediction):
        """
        Performs backpropagation by calculating gradients and updating weights and bias.

        Parameters:
        -----------
        y_prediction : np.ndarray
            The predicted values from the forward pass.
        """
        # Calculate gradient of the loss with respect to the weight
        w_grad = (2 / self.N) * np.dot((y_prediction - self.y_values), self.x_values)
        # Calculate gradient of the loss with respect to the bias
        b_grad = (2 / self.N) * np.sum(y_prediction - self.y_values)

        # Update weight and bias using gradient descent
        self.w -= self.alpha * w_grad
        self.b -= self.alpha * b_grad

    def train(self, epochs: int, showProgression=True, numberOfProgressionPoints=10):
        """
        Trains the model using gradient descent for a specified number of epochs.

        Parameters:
        -----------
        epochs : int
            The number of iterations to run the gradient descent.
        showProgression : bool, optional
            If True, prints the loss, weight, and bias at specified progression points (default is True).
        numberOfProgressionPoints : int, optional
            The number of progression points to display during training (default is 10).
        """
        if showProgression:
            progression_points = np.linspace(0, epochs-1, numberOfProgressionPoints, dtype=int)

        for epoch in range(epochs):
            # Perform a forward pass
            y_prediction = self.forward()
            # Perform a backward pass (backpropagation)
            self.back(y_prediction)

            # Print progress at specified intervals
            if showProgression and epoch in progression_points:
                loss = self.loss(y_prediction)
                print(f"After {epoch+1} epochs, w = {self.w}, b = {self.b}, loss = {loss}")

    def __call__(self, x_input):
        """
        Allows the model to be called as a function to make predictions.

        Parameters:
        -----------
        x_input : float or np.ndarray
            The input data to predict values for.

        Returns:
        --------
        np.ndarray
            The predicted values.
        """
        return self.w * x_input + self.b
    
    def plot(self, max_points=100):
        """
        Plots the regression line and the scatter plot of the data points.

        Parameters:
        -----------
        max_points : int, optional
            The maximum number of data points to plot (default is 100).
        """
        # Limit the number of data points plotted
        if len(self.x_values) > max_points:
            indices = np.random.choice(len(self.x_values), max_points, replace=False)
            x_values_sample = self.x_values[indices]
            y_values_sample = self.y_values[indices]
        else:
            x_values_sample = self.x_values
            y_values_sample = self.y_values

        # Create the scatter plot of the data points
        plt.scatter(x_values_sample, y_values_sample, label='Data Points', color='blue')

        # Plot the regression line
        x_min, x_max = np.min(self.x_values), np.max(self.x_values)
        x_range = np.linspace(x_min, x_max, 100)
        y_range = self.w * x_range + self.b
        plt.plot(x_range, y_range, label='Regression Line', color='red')

        # Add labels and legend
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression Model')
        plt.legend()

        # Show the plot
        plt.show()



## Functions called in main()

def test_data():
    # Parameters for data generation
    w = 2.0
    b = 1.0
    sigma = 0.5
    interval = np.array([0, 10])
    numDatapoints = 50

    # Generate the data
    x_values, y_values = generate_data(w, b, sigma, interval, numDatapoints)

    # Plot the generated data points
    plt.scatter(x_values, y_values, label='Generated Data', color='blue')

    # Plot the theoretical line
    x_theoretical = np.linspace(interval[0], interval[1], 100)
    y_theoretical = w * x_theoretical + b
    plt.plot(x_theoretical, y_theoretical, label='Theoretical Line', color='red')

    # Labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Generated Data and Theoretical Line')
    plt.legend()

    # Show the plot
    plt.show()

def test_linear_regression_model(x_values, y_values, alpha, epochs):
    Model = LinearRegressionMLModel(x_values, y_values, alpha)
    Model.train(epochs)
    w_analytical, b_analytical = calculate_analytical_solution(Model.x_values, Model.y_values)
    print("")
    print(f"weight and bias from ML after {epochs} epochs: w = {Model.w}, b = {Model.b}")
    print(f"weight and bias from normal equation: w = {w_analytical}, b = {b_analytical}")

    Model.plot()





def main():
    #test_data()
    x_values, y_values = generate_data(2, 1, 0.1, np.array([0,10]), 1000)
    test_linear_regression_model(x_values, y_values, 0.01, 2000)

if __name__ == "__main__":
    main()