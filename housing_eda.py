import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    """Load the California Housing dataset."""
    data = fetch_california_housing()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    return data, df


def show_basic_info(df):
    """Show the first rows, shape and missing values."""
    print(df.head())
    print("\nShape:", df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())


def show_summary(df):
    """Show summary statistics."""
    print("\nSummary Statistics:")
    print(df.describe())


def show_correlation(df):
    """Show correlation with the target."""
    print("\nCorrelation with target:")
    print(df.corr()["target"].sort_values(ascending=False))


def plot_histograms(df):
    """Plot histograms of the dataset."""
    df.hist(figsize=(12, 10))
    plt.tight_layout()
    plt.savefig("figures/histograms.png")
    plt.show()


def plot_correlation(df):
    """Plot the correlation matrix."""
    corr = df.corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm")
    plt.colorbar()

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=90
    )

    plt.yticks(
        range(len(corr.columns)),
        corr.columns
    )

    plt.title("Correlation Matrix")
    plt.tight_layout()

    plt.savefig("figures/correlation_matrix.png")
    plt.show()


def plot_location(df):
    """Plot location and target values."""
    plt.figure(figsize=(8, 6))

    plt.scatter(
        df["Longitude"],
        df["Latitude"],
        c=df["target"],
        s=10
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("California Housing Prices")

    plt.colorbar(label="Target")

    plt.savefig("figures/location_target.png")
    plt.show()


def check_data_leakage(df, feature_names):
    """Compare correct and incorrect standardization methods."""
    X = df[feature_names]

    X_train, X_test = train_test_split(
        X,
        test_size=0.2,
        random_state=42
    )

    # Method 1: training data only
    scaler1 = StandardScaler()

    X_train_1 = scaler1.fit_transform(X_train)
    X_test_1 = scaler1.transform(X_test)

    # Method 2: whole dataset
    scaler2 = StandardScaler()

    X_all_2 = scaler2.fit_transform(X)

    X_train_2 = X_all_2[X_train.index]
    X_test_2 = X_all_2[X_test.index]

    print("\nTest means - Method 1:")
    print(X_test_1.mean(axis=0))

    print("\nTest means - Method 2:")
    print(X_test_2.mean(axis=0))

    print("\nDifference:")
    print(X_test_1.mean(axis=0) - X_test_2.mean(axis=0))


def main():
    """Run the California Housing analysis."""
    data, df = load_data()

    show_basic_info(df)
    show_summary(df)
    show_correlation(df)

    plot_histograms(df)
    plot_correlation(df)
    plot_location(df)

    check_data_leakage(df, data.feature_names)


if __name__ == "__main__":
    main()