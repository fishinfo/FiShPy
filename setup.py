from distutils.core import setup
setup(
  name = 'FiShPy',         
  packages = ['FiShPy'],   
  version = '1.0',      
  license='MIT',        
  description = 'Fisher-Shannon method (uncomplete work)', 
  author = 'Fabian Guignard',               
  author_email = 'fabian.guignard@bluemail.ch',    
  url = 'https://github.com/fishinfo/FiShPy', 
  download_url = 'https://github.com/fishinfo/FiShPy/archive/v_0.1.tar.gz',   
  keywords = [
      'Fisher Shannon plane', 
      'Fisher information measure', 
      'Fisher-Shannon complexity', 
      'Statistical complexity',
      ],  
  install_requires=[     
          'numpy',
          'scipy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha', 
    'Topic :: Software Development :: Build Tools',    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3.7',
    ],
)
