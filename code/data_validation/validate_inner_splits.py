from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_inner_splits import main
if __name__ == '__main__':
    sys.argv.insert(1, 'validate')
    main()
