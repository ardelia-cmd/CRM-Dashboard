# db_utils.py

import os
from dotenv import load_dotenv
from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine

load_dotenv()  # load .env file

SSH_HOST = os.environ.get("SSH_HOST")
SSH_PORT = int(os.environ.get("SSH_PORT", 22))
SSH_USER = os.environ.get("SSH_USER")
SSH_PASSWORD = os.environ.get("SSH_PASSWORD")

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", 5432))
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")


def create_tunnel_and_engine():
    server = SSHTunnelForwarder(
        (SSH_HOST, SSH_PORT),
        ssh_username=SSH_USER,
        ssh_password=SSH_PASSWORD,
        remote_bind_address=(DB_HOST, DB_PORT),
    )
    server.start()

    local_port = server.local_bind_port
    db_url = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
        f"@127.0.0.1:{local_port}/{DB_NAME}"
    )
    engine = create_engine(db_url)
    return engine, server



def create_tunnel_and_engine():
    server = SSHTunnelForwarder(
        (SSH_HOST, SSH_PORT),
        ssh_username=SSH_USER,
        ssh_password=SSH_PASSWORD,
        remote_bind_address=(DB_HOST, DB_PORT),
    )
    server.start()

    local_port = server.local_bind_port
    db_url = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
        f"@127.0.0.1:{local_port}/{DB_NAME}"
    )
    engine = create_engine(db_url)
    return engine, server
