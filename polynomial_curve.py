import numpy as np
import matplotlib.pyplot as plt


def make_data(n):
    """Create the data."""
    x = np.random.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.3, n)
    return x, y


def fit_polynomial(x, y, degree):
    """Fit a polynomial."""
    p = np.polyfit(x, y, degree)
    return p


def rmse(y, predicted):
    """Calculate RMSE."""
    return np.sqrt(np.mean((y - predicted) ** 2))


def main():
    """Run the polynomial experiments."""
    np.random.seed(42)

    x_train, y_train = make_data(100)
    x_test, y_test = make_data(1000)

    print("Training data:", len(x_train))
    print("Test data:", len(x_test))

    # Polynomial fits
    degrees = [0, 1, 3, 9]

    x_line = np.linspace(0, 1, 1000)
    y_true = np.sin(2 * np.pi * x_line)

    plt.figure(figsize=(10, 8))

    for i, degree in enumerate(degrees):
        p = fit_polynomial(x_train, y_train, degree)
        y_fit = np.polyval(p, x_line)

        plt.subplot(2, 2, i + 1)
        plt.scatter(x_train, y_train, label="Training data")
        plt.plot(x_line, y_true, label="True function")
        plt.plot(x_line, y_fit, label="Fitted polynomial")

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Degree " + str(degree))
        plt.legend()

    plt.tight_layout()
    plt.savefig("figures/polynomial_fits.png")
    plt.show()

    # RMSE
    train_rmse = []
    test_rmse = []

    for degree in range(13):
        p = fit_polynomial(x_train, y_train, degree)

        train_pred = np.polyval(p, x_train)
        test_pred = np.polyval(p, x_test)

        train_rmse.append(rmse(y_train, train_pred))
        test_rmse.append(rmse(y_test, test_pred))

    print("Training RMSE:", train_rmse)
    print("Test RMSE:", test_rmse)

    print("Best test degree:", np.argmin(test_rmse))
    print("Best training degree:", np.argmin(train_rmse))

    plt.figure()
    plt.plot(range(13), train_rmse, marker="o", label="Training RMSE")
    plt.plot(range(13), test_rmse, marker="o", label="Test RMSE")

    plt.xlabel("Polynomial Degree")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Training and Test RMSE")

    plt.savefig("figures/rmse.png")
    plt.show()


if __name__ == "__main__":
    main()