class lhl_delay_data:
    """
    Pull from database for Lighthouse Labs Midterm Project
    """
    def __init__ (self, host, database, user, password):
        self.host = host
        self.database = database
        self.user = user
        self.password = password

    def connection(self):
        return psycopg2.connect(
            host = self.host,
            database = self.database,
            user = self.user,
            password = self.password
        )

    def sample_table(self, table: str, samples:int):
        if table not in ['flights','passengers','fuel_comsumption']:
            print('Error: {table} is not recognized as a table name.')

        cur = self.connection().cursor()
        query = f"""
        SELECT *
        FROM {table}
        ORDER BY 
        """
        cur.execute()
        
        pass

    def close_conn (self):
        self.conn.close()

    def __doc__(self):
        pass
