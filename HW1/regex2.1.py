import re

# Match all emails in text and compile a set of all found email addresses.

with open("51060", encoding="utf8", errors='ignore') as infile:
    potentialEmails = infile.read()

emails = re.findall(r'[\w.-]+@[\w\.]+\.[\w]+', potentialEmails)
print(emails)
