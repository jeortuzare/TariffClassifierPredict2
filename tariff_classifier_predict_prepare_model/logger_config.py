import logging


def setup_logging():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename='app.log',
                        filemode='w')  # 'w' para sobrescribir el archivo en cada ejecución, 'a' para añadir al final

    # Opción para agregar un manejador de consola además del archivo
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # Obtener el logger raíz y agregar el manejador de consola
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
