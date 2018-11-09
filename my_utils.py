import os
import shutil

def prepare_error_output(error_directory):
    if not os.path.exists('./errors'):
        os.makedirs('errors')

    if not os.path.exists("./errors/%s" % (error_directory)):
        os.makedirs("./errors/%s" % (error_directory))

    shutil.rmtree("./errors/%s" % (error_directory))
    os.makedirs("./errors/%s" % (error_directory))