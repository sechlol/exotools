from exotools.datasets.planetary_composite import PlanetarySystemsCompositeDataset
from exotools.utils.warning_utils import warnings_as_exceptions


def main():
    ds = PlanetarySystemsCompositeDataset()
    db = ds.download_composite_dataset(limit=10)
    print(len(db))
    print(db.to_pandas())


if __name__ == "__main__":
    with warnings_as_exceptions():
        main()
