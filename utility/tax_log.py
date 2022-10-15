import logging

log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
log.addHandler(console_handler)

file_handler = logging.FileHandler('tax.log')
file_handler.setFormatter(log_formatter)
log.addHandler(file_handler)