import datetime


def get_timestamp():
    return str(datetime.datetime.now(datetime.timezone.utc))