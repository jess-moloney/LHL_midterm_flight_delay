from modules import connect_lhl, test_models, pipelines, constants
import pickle


def main():
    with open(r"..\..\data\flight_data_engineered.pickle", "rb") as flight_data_file:
        df_test = pickle.load(flight_data_file)
    pass


if __name__ is "__main__":
    main()
