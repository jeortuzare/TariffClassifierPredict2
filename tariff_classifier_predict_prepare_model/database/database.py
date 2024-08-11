import sys

import psycopg2
import pandas as pd
from psycopg2 import sql
import logging
from datetime import date
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, dbname, user, password, host, port=None):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = None
        self.cursor = None
        self.engine = None
        self.sections_index = {}
        self._tariff_description_dict = {}
        self.get_sections_index()
        self.chapter_count_by_section = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
            )
            if self.port is not None:
                self.conn['port'] = self.port
            self.cursor = self.conn.cursor()
            self.engine = create_engine('postgresql+psycopg2://', creator=lambda: self.conn)
            logger.info("Conexión exitosa a la base de datos")
        except Exception as e:
            logger.error(f"Ocurrió un error al conectar a la base de datos: {e}")

    def disconnect(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Conexión a la base de datos cerrada")

    def insert_commodity(self, description, tariff_classification, year, month, doc_number, doc_date, doc_item,
                         doc_type):
        try:
            if self.conn is None or self.cursor is None:
                self.connect()

            insert_query = sql.SQL("""
                INSERT INTO commodity (description, tariff_classification, year, month, doc_number, doc_date, doc_item, doc_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """)

            self.cursor.execute(insert_query,
                                (description, tariff_classification, year, month, doc_number, doc_date, doc_item,
                                 doc_type))
            self.conn.commit()


        except Exception as e:
            logger.error(f"Ocurrió un error al insertar el registro: {e}")
            if self.conn:
                self.conn.rollback()

    def insert_tariff(self, index, code, description, t_u, value, s_u, tab, level, code_key=None):
        try:
            if self.conn is None or self.cursor is None:
                self.connect()

            insert_query = sql.SQL("""
                INSERT INTO tariff (index, code, description, t_u, value, s_u, tab, level, code_key)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """)

            self.cursor.execute(insert_query,
                                (index, code, description, t_u, value, s_u, tab, level, code_key))
            self.conn.commit()


        except Exception as e:
            logger.error(f"Ocurrió un error al insertar el registro: {e}")
            if self.conn:
                self.conn.rollback()
            self.disconnect()
            raise e

    def __del__(self):
        self.disconnect()

    def exec_query(self, query_sql, return_rows=False):
        if self.conn is None:
            self.connect()
        self.chapter_count_by_section = {}
        query = sql.SQL(query_sql)
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query)
                if return_rows:
                    return cursor.fetchall()
                else:
                    return cursor
        except Exception as e:
            logger.inerrorfo(f"Error al ejecutar la consulta: {e}")
            raise e


    def get_tariff(self, code):
        """
        Selecciona un registro de la tabla 'tariff' basado en el campo 'code'.

        :param code: El código del registro a seleccionar.
        :return: Un diccionario con los campos del registro seleccionado o None si no se encuentra.
        """
        if self.conn is None:
            self.connect()

        query = sql.SQL("SELECT index, code, description, t_u, value, s_u, tab, level FROM tariff WHERE code = %s")

        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, (code,))
                row = cursor.fetchone()
                if row:
                    return {
                        "index": row[0],
                        "code": row[1],
                        "description": row[2],
                        "t_u": row[3],
                        "value": row[4],
                        "s_u": row[5],
                        "tab": row[6],
                        "level": row[7]
                    }
                else:
                    logger.info(f"No se encontró ningún registro con el código proporcionado code:{code}.")
                    return False
        except Exception as e:
            logger.inerrorfo(f"Error al ejecutar la consulta: {e}")
            raise e

    def get_sections_index(self):
        if self.conn is None:
            self.connect()
        query = sql.SQL("SELECT code, section  FROM chapter order by code")
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                for row in rows:
                    self.sections_index[row[0]]= row[1]
        except Exception as e:
            logger.error(f"Error al ejecutar la consulta: {e}")
            raise e

    def get_chapter_count_by_section(self):
        if self.chapter_count_by_section is not None:
            return self.chapter_count_by_section
        if self.conn is None:
            self.connect()
        self.chapter_count_by_section = {}
        query = sql.SQL("SELECT section, count(*)  FROM chapter group by section order by section")
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                for row in rows:
                    self.chapter_count_by_section[row[0]]= row[1]
        except Exception as e:
            logger.error(f"Error al ejecutar la consulta: {e}")
            raise e
        return self.chapter_count_by_section

    def get_chapters_by_section(self, section):
        sections_index = list(range(21))
        if section not in sections_index:
            raise Exception(f'Section f{section} invalid!!!')
        if self.conn is None:
            self.connect()
        query = sql.SQL(f"SELECT code, description FROM chapter where section = {section} order by code")
        out= {}
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                for row in rows:
                    out[row[0]]= row[1]
        except Exception as e:
            logger.error(f"Error al ejecutar la consulta: {e}")
            raise e
        return out

    def get_chapter_index(self, group=None):
        if group is None:
            index_range = range(1, 98)
        elif group == 1:
            index_range = range(1, 26)
        elif group == 2:
            index_range = range(26, 51)
        elif group == 3:
            index_range = range(51, 76)
        elif group == 4:
            index_range = range(76, 85)
        elif group == 5:
            index_range = range(85, 86)
        elif group == 6:
            index_range = range(86, 98)
        else:
            raise Exception(f'no valid group {group}!!')
        chapter_index = []
        for int_chapter in index_range:
            if int_chapter < 10:
                chapter = '0' + str(int_chapter)
            else:
                chapter = str(int_chapter)
            chapter_index.append(chapter)
        return chapter_index

    def get_positions_by_chapter(self, chapter):
        chapter_index = self.get_chapter_index()
        if chapter not in chapter_index:
            raise Exception(f'Chapter f{chapter} invalid!!!')
        if self.conn is None:
            self.connect()
        query = sql.SQL(f"select distinct(code_key) from tariff where code_key like '{chapter}%' order by code_key")
        out = []
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                for row in rows:
                    out.append(row[0])
        except Exception as e:
            logger.error(f"Error al ejecutar la consulta: {e}")
            raise e
        return out

    def get_sections_codes(self):
        if self.conn is None:
            self.connect()
        out = {}
        query = sql.SQL("SELECT code, id  FROM section order by id")
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                for row in rows:
                    out[row[0]]= row[1]
        except Exception as e:
            logger.error(f"Error al ejecutar la consulta: {e}")
            raise e
        return out

    def get_sections_ids(self):
        if self.conn is None:
            self.connect()
        out = {}
        query = sql.SQL("SELECT code, id, description FROM section order by id")
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                for row in rows:
                    out[row[1]] = (row[0], row[2])
        except Exception as e:
            logger.error(f"Error al ejecutar la consulta: {e}")
            raise e
        return out

    def test_get_tariff_description(self):
        logger.info(f"testing tariff descriptions")
        if self.conn is None:
            self.connect()
        query = sql.SQL("""select distinct tariff_classification, doc_date from commodity 
                           where tariff_classification  not in  ('0nan', 'nan') order by 1""")
        with self.conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            for row in rows:
                logger.info(f"checking {row[0]} {row[1]}")
                if not self.get_tariff_description(code_key=row[0], valid_date=row[1]):
                    raise Exception(f"No se encuentra clasificacion para {row[0]} {row[1]}")

    def get_tariff_description(self, code_key, format=False, valid_date=None):
        """
        Selecciona un registro de la tabla 'tariff' basado en el campo 'code'.

        :param code: El código del registro a seleccionar.
        :return: Un diccionario con los campos del registro seleccionado o None si no se encuentra.
        """
        if valid_date is None:
            valid_date = date.today()
        if code_key.endswith('.0'):
            code_key = code_key[:-2]
        if len(code_key) == 7:
            code_key = '0' + code_key
        if len(code_key) == 6:
            code_key = '00' + code_key
        if code_key in self._tariff_description_dict:
            if not format:
                return self._tariff_description_dict[code_key]
            else:
                return self.format_description(self._tariff_description_dict[code_key])
        if self.conn is None:
            self.connect()
        query = sql.SQL(
            """SELECT index, code, description, t_u, value, s_u, tab, level 
            FROM tariff ta
            WHERE ta.index <= (select t.index 
                                from tariff  t 
                                where t.code_key = %s 
                                and t.valid_from <= %s and (t.valid_to >= %s or t.valid_to is null))
            and ta.valid_from <= %s and (ta.valid_to >= %s or ta.valid_to is null) 
            order by index desc """)

        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, (code_key, valid_date, valid_date, valid_date, valid_date))
                rows = cursor.fetchall()
                if len(rows) > 0:
                    list_out = []
                    for row in rows:
                        list_out.append({
                            "index": row[0],
                            "code": row[1],
                            "description": row[2],
                            "t_u": row[3],
                            "value": row[4],
                            "s_u": row[5],
                            "tab": row[6],
                            "level": row[7]
                        })
                        if row[7] == 0:
                            list_out.reverse()
                            key0 = code_key[0:2]
                            key1 = code_key[0:2] + '.' + code_key[2:4]
                            key2 = code_key[0:4] + '.' + code_key[4:6]
                            key3 = code_key[0:4] + '.' + code_key[4:6] + '00'
                            key4 = code_key[0:4] + '.' + code_key[4:8]

                            out = []
                            for record in list_out:
                                if record['code'] in (key0, key1, key2, key3, key4):
                                    out.append(record)
                            last_record = out[len(out) - 2]
                            sub_partidas = []
                            for record in list_out:
                                if record['code'] is None and record['index'] > last_record['index']:
                                    sub_partidas.append(record)
                            if len(sub_partidas) > 0:
                                sub_partida = sub_partidas[len(sub_partidas) - 1]
                                out.insert(len(out) - 1, sub_partida)
                            self._tariff_description_dict[code_key] = out
                            if not format:
                                return out
                            else:
                                return self.format_description(out)
                else:
                    logger.info(f"No se encontró ningún registro con el código proporcionado code_key:{code_key}.")
                    return False
        except Exception as e:
            logger.error(f"Error al ejecutar la consulta: {e}")
            raise e

    def get_all_tariff(self):
        if self.conn is None:
            self.connect()

        query = sql.SQL(
            """SELECT index, code, description, t_u, value, s_u, tab, level, code_key
            FROM tariff 
            order by index""")

        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                if len(rows) > 0:
                    out = []
                    for row in rows:
                        out.append({
                            "index": row[0],
                            "code": row[1],
                            "description": row[2],
                            "t_u": row[3],
                            "value": row[4],
                            "s_u": row[5],
                            "tab": row[6],
                            "level": row[7],
                            "code_key": row[8],
                        })
                    return out
                else:
                    logger.info("No se encontró registro en la tabla tariff.")
                    return False
        except Exception as e:
            logger.error(f"Error al ejecutar la consulta: {e}")
            raise e

    def get_comodities_by_chapter(self, chapter, offset, limit):
        if self.conn is None:
            self.connect()
        query = f"SELECT  description as commodity_description, tariff_classification, doc_date " \
                f"FROM commodity " \
                f"WHERE tariff_classification like '{chapter}%' " \
                f"order by doc_date LIMIT {limit} OFFSET {offset}"
        df = pd.read_sql_query(query, self.engine)
        df['commodity_chapter'] = None
        for index, row in df.iterrows():
            tariff_classification = row['tariff_classification']
            doc_date = row['doc_date']
            commodity_chapter = tariff_classification[:2]
            df.at[index, 'commodity_chapter'] = commodity_chapter
        return df

    def format_description(self, description):
        lineas = ''
        for record in description:
            code = record['code']
            tab = record['tab']
            description = record['description']
            if code is None:
                code = ''
            if tab is None:
                tab = ''
            else:
                tab += ' '
            linea = code + '\t' + tab + description + '\n'
            lineas += linea
        return lineas

    def get_num_classes(self):
        if self.conn is None:
            self.connect()

        query = sql.SQL(
            """SELECT count(*)
            FROM tariff 
            where code_key is not null""")

        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query)
                row = cursor.fetchone()
                return row[0]
        except Exception as e:
            logger.error(f"Error al ejecutar la consulta: {e}")
            raise e

    def insert_correlation(self, old_tariff_index, old_tariff_valid_from, new_tariff_index, new_tariff_valid_from):
        try:
            insert_query = sql.SQL("""
                            INSERT INTO correlation (old_valid_from, old_index, new_valid_from, new_index)
                            VALUES (%s, %s, %s, %s)
                        """)

            self.cursor.execute(insert_query,
                                (old_tariff_valid_from, old_tariff_index, new_tariff_valid_from, new_tariff_index))
            self.conn.commit()
        except Exception as e:
            print(f"Ocurrió un error al insertar el registro: {e}")
            if self.conn:
                self.conn.rollback()

    def get_comodities(self, offset=None, limit=None, with_tariff_classification_description=True, with_chapter=False,
                       with_section=False, year=None, month=None, order_by='doc_date', chapter_zero=False, section=None,
                       chapter=None):
        if self.conn is None:
            self.connect()

        query = f"SELECT  description as commodity_description, tariff_classification, doc_date " \
                f"FROM commodity "
        if chapter_zero:
            query += "WHERE tariff_classification <= '00999999' "
        else:
            query += "WHERE tariff_classification > '00999999' "
        if section is not None:
            chapters= self.get_chapters_by_section(section).keys()
            min_class = min(chapters) + '000000'
            max_class = max(chapters) + '999999'
            query += f"AND tariff_classification >= '{min_class}' "
            query += f"AND tariff_classification <= '{max_class}' "
        if chapter is not None:
            min_class = chapter + '000000'
            max_class = chapter + '999999'
            query += f"AND tariff_classification >= '{min_class}' "
            query += f"AND tariff_classification <= '{max_class}' "
        if year is not None:
            query += f"and year = {year} "
        if month is not None:
            query += f"and month = {month} "
        query += f"order by {order_by} "
        if limit is not None:
            query += f"LIMIT {limit} "
        if offset is not None:
            query += f"OFFSET {offset}"
        #print(query)
        df = pd.read_sql_query(query, self.engine)
        if with_tariff_classification_description:
            df['tariff_classification_description'] = None
            for index, row in df.iterrows():
                tariff_classification = row['tariff_classification']
                doc_date = row['doc_date']
                tariff_classification_description = self.get_tariff_description(tariff_classification, format=True,
                                                                                valid_date=doc_date)
                if not tariff_classification_description:
                    raise Exception(
                        f"Error de inconsistencia, no se encuentra descrpción para {row['tariff_classification']}")
                df.at[index, 'tariff_classification_description'] = tariff_classification_description
        if with_chapter or with_section:
            if with_chapter:
                df['commodity_chapter'] = None
            if with_section:
                df['commodity_section'] = None
            for index, row in df.iterrows():
                tariff_classification = row['tariff_classification']
                doc_date = row['doc_date']
                commodity_chapter = tariff_classification[:2]
                if with_chapter:
                    df.at[index, 'commodity_chapter'] = commodity_chapter
                if with_section:
                    df.at[index, 'commodity_section'] = self.sections_index[commodity_chapter]
        return df