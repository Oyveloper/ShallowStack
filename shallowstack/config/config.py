import configparser

config = configparser.ConfigParser()
config.read("config.ini")

POKER_CONFIG: configparser.SectionProxy = config["POKER"]
RESOLVER_CONFIG: configparser.SectionProxy = config["RESOLVER"]
