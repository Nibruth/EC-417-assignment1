import numpy as np
import matplotlib.pyplot as plt


def make_data():
    """Create one training dataset."""
    x = np.random.uniform(0, 1, 15)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.3, 15)
    return x, y


def get_predictions(degree, x_test):
    """Train 30 models and get their predictions."""
    predictions = []

    for i in range(30):
        x_train, y_train = make_data()

        p = np.polyfit(x_train, y_train, degree)
        y_pred = np.polyval(p, x_test)

        predictions.append(y_pred)

    return np.array(predictions)


def calculate_bias_variance(predictions, y_true):
    """Calculate bias squared and variance."""
    average = np.mean(predictions, axis=0)

    bias = np.mean((average - y_true) ** 2)
    variance = np.mean((predictions - average) ** 2)

    return bias, variance


def main():
    """Run the bias-variance experiment."""
    np.random.seed(42)

    x_test = np.random.uniform(0, 1, 1000)
    y_true = np.sin(2 * np.pi * x_test)

    degrees = [0, 3, 9]
    models = {}

    for degree in degrees:
        models[degree] = get_predictions(degree, x_test)

    print("Degree 0:", models[0].shape)
    print("Degree 3:", models[3].shape)
    print("Degree 9:", models[9].shape)

    bias_values = []
    variance_values = []
    error_values = []

    for degree in degrees:
        predictions = models[degree]

        bias, variance = calculate_bias_variance(
            predictions, y_true
        )

        test_error = bias + variance + 0.3 ** 2

        bias_values.append(bias)
        variance_values.append(variance)
        error_values.append(test_error)

        print("\nDegree:", degree)
        print("Bias squared:", bias)
        print("Variance:", variance)
        print("Expected test error:", test_error)
        print("Bias squared + Variance + Noise:",
              bias + variance + 0.3 ** 2)

    plt.plot(
        degrees,
        bias_values,
        marker="o",
        label="Bias squared"
    )

    plt.plot(
        degrees,
        variance_values,
        marker="o",
        label="Variance"
    )

    plt.plot(
        degrees,
        error_values,
        marker="o",
        label="Test error"
    )

    plt.xlabel("Polynomial Degree")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Bias-Variance Decomposition")

    plt.savefig("figures/bias_variance.png")
    plt.show()


if __name__ == "__main__":
    main()