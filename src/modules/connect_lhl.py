import psycopg2
import pandas.io.sql as sqlio

class lhl_delay_data:
    """
    Access and pull from database for Lighthouse Labs Midterm Project
    """
    
    def __init__ (self, host, database, user, password, seed = None):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.seed = seed

    def connection(self):
        return psycopg2.connect(
            host = self.host,
            database = self.database,
            user = self.user,
            password = self.password
        )

    def sample_table(self, table: str, samples:int):
        """
        Randomly sample from LHL Database for the midterm project
        :params:
        table (str): Which of the three main tables
                    ('flights','passengers','fuel_comsumption','flights_test')
                    to pull from
        samples (int): max samples to pull
                        (limited by number of samples in each table)

        :return:
        df_return (pandas.DataFrame): DataFrame containing the selected samples
        """
        
        if table not in ['flights','passengers','fuel_comsumption','flights_test']:
            print(f'Error: {table} is not recognized as a table name.')

        cur = self.connection().cursor()
        query = ''
        if self.seed is not None:
            query += f'SELECT SETSEED({self.seed}); '
        query += f'SELECT * FROM {table} '
       
        if table != 'flights' or table != 'flights_test':
            query += f'WHERE {table}.unique_carrier IN (SELECT op_unique_carrier FROM flights) '
        query += f'ORDER BY RANDOM() LIMIT {samples};'

        df_return = sqlio.read_sql_query(query, self.connection())
        
        return df_return