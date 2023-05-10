base_url=https://raw.githubusercontent.com/python/cpython/3.9/Lib

files=("__init__.py" \
	   "__main__.py" \
	   "_uninstall.py" \
	   "_bundled/__init__.py" \
	   "_bundled/pip-23.0.1-py3-none-any.whl" \
	   "_bundled/setuptools-58.1.0-py3-none-any.whl")

for _f in ${files[@]}; do

    f=ensurepip/${_f}

    if test ! -f "$f"; then
        wget -q --show-progress ${base_url}/${f} -P $(dirname $f);
	else
		echo -e "'${f}' already exists. Nothing to do."
    fi
done
