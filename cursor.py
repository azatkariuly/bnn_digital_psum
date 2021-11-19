import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'progress'))
from progress.bar import Bar as Bar

bar = Bar('Processing', max=1)
bar.suffix  = '{phase}'.format(phase='Cursome, come back!')
bar.finish()
