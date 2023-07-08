mkdir _dist
cp -r {docs_skeleton,snippets} _dist
mkdir -p _dist/docs_skeleton/static/api_reference
cd api_reference
poetry run make html
cp -r _build/* ../_dist/docs_skeleton/static/api_reference
cd ..
cp -r extras/* _dist/docs_skeleton/docs
cd _dist/docs_skeleton
poetry run nbdoc_build
yarn install
yarn start
