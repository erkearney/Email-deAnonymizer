""" Thanks to these Stackoverflow threads for helping me write this:
https://stackoverflow.com/questions/123198/how-do-i-copy-a-file-in-python
https://stackoverflow.com/questions/8933237/how-to-find-if-directory-exists-in-python
https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
"""
import os, os.path
import shutil
from shutil import copyfile
from shutil import rmtree

datasetLocation = "F:/Machine Learning/enron_mail_20150507.tar/maildir/"
fileDestination = "F:/Machine Learning/enron_mail_20150507.tar/sent mail/"

minEmail = 40

def moveAndRenameFiles(datasetLocation, fileDestination):
    for user in os.listdir(datasetLocation):
        if os.path.exists(datasetLocation + user + "/sent"):
        # Some of the users, for whatever reason, don't have a 'sent' folder in their directory
            os.makedirs(fileDestination + user)
            # Make a new directory for each user, this directory will contain only the files from their original sent folder
            for file in os.listdir(datasetLocation + user + "/sent"):
                prevName = datasetLocation + user + '/sent/' + file
                newName = fileDestination + user + '/' + user + file + ".txt"
                # Add the .txt file extension to each file so we can read it
                
                copyfile(prevName, newName)
        else:
            print(user + " has no sent folder")

def cleanFiles(fileDestination):
    # Remove users that have too few emails
    for user in os.listdir(fileDestination):
        print(user, " has ", len([name for name in os.listdir(fileDestination + '/' + user) if os.path.isfile(os.path.join(fileDestination  + '/' + user, name))]), " emails.")
        if len([name for name in os.listdir(fileDestination + '/' + user) if os.path.isfile(os.path.join(fileDestination  + '/' + user, name))]) < minEmail:
            print(user, " has too few emails, deleting. . .")
            shutil.rmtree(fileDestination + '/' + user)

    # Remove unnecessary metadata such as date sent, recipients, whitespace, etc. from emails
    file = open(fileDestination, "r")
    lines = file.readlines()
    file.close()

    file = open(fileDestination, "w")

    deleting = True

    for line in lines:
        line.strip()
        # I'm actually not sure why line.strip() here removes blank lines, but, it works. . .
        if deleting == False and line.strip(): file.write(line)
        # Every emails meta ends with "X-FileName: " and the sender's name
        # So once we reach this line, we switch the 'deleting' boolean to False
        if "X-FileName: " in line: deleting = False

    file.close()

moveAndRenameFiles(datasetLocation, fileDestination)

# Clean the data!
for user in os.listdir(fileDestination):
    for email in os.listdir(fileDestination + user + "/"):
        cleanFiles(fileDestination + user + "/" + email)
