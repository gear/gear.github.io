#!/root/anaconda3/bin/python

# Python 3.5.2
# utf-8
# Convert github link to raw link

from sys import argv
from urllib.parse import urlparse

def main():
    """
    Main function to convert and print new link to console.
    """
    assert len(argv) == 2, "Usage: to_rawgit 'https://github.com/...'"
    git_link = argv[1]
    print('Link to convert:' + git_link)
    parsed = urlparse(git_link)
    content = parsed.path.split('/')
    content.remove('blob')
    new_path = '/'.join(content)
    new_link = parsed._replace(netloc='cdn.rawgit.com', path=new_path)
    print(new_link.geturl()) 

if __name__ == '__main__':
    main()
