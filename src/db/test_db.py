from src.db.database import test_connection


def main():
    database_name = test_connection()
    print("Connected successfully.")
    print(f"Current database: {database_name}")


if __name__ == "__main__":
    main()