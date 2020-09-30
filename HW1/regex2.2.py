import re

# Find all dates in text (e.g. 04/12/2019, April 20th 2019, etc).

with open("51060", encoding="utf8", errors='ignore') as infile:
    potentialDates = infile.read()

dates_slashes = re.findall(r'[\d]\d?\/[\d]\d?\/[\d][\d]\d?\d?', potentialDates)
dates_verbose_a = re.findall(r'[a-zA-Z]+\s[\d]\d?[a-zA-Z]?[a-zA-Z]?,?\s[\d][\d][\d][\d]', potentialDates)
dates_verbose_b = re.findall(r'[\d]\d?[a-zA-Z]?[a-zA-Z]?\s[a-zA-Z]+\s[\d][\d][\d][\d]', potentialDates)
dates_dashes = re.findall(r'[\d]\d?-[\d]\d?-[\d][\d]\d?\d?', potentialDates)
all_dates = dates_slashes + dates_verbose_a + dates_verbose_b + dates_dashes
print(all_dates)