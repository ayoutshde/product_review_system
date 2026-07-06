from src.db.database import engine, Base
from src.db import models


def main():
    print("Creating PostgreSQL tables...")
    Base.metadata.create_all(bind=engine)
    print("Done. Tables created successfully.")


if __name__ == "__main__":
    main()