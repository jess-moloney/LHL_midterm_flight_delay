import psycopg2
import pandas.io.sql as sqlio

class lhl_delay_data:
    """
    Pull from database for Lighthouse Labs Midterm Project
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
        
        if table not in ['flights','passengers','fuel_comsumption']:
            print(f'Error: {table} is not recognized as a table name.')

        cur = self.connection().cursor()
        query = ''
        if self.seed is not None:
            query += f'SELECT SETSEED({self.seed}); '
        query += f'SELECT * FROM {table} '
       
        if table != 'flights':
            query += f'WHERE {table}.unique_carrier IN (SELECT op_unique_carrier FROM flights) '
        query += f'ORDER BY RANDOM() LIMIT {samples};'

        df_return = sqlio.read_sql_query(query, self.connection())
        
        return df_return