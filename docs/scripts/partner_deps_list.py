import yaml

if __name__ == "__main__":
    with open("../libs/packages.yml", "r") as f:
        packages_yaml = yaml.safe_load(f)

    packages = packages_yaml["packages"]

    comaintain_packages = [
        p
        for p in packages
        if not p.get("disabled", False)
        and p["repo"].startswith("langchain-ai/")
        and p["repo"] != "langchain-ai/langchain"
    ]
    monorepo_packages = [
        p
        for p in packages
        if not p.get("disabled", False) and p["repo"] == "langchain-ai/langchain"
    ]

    for p in monorepo_packages:
        print("--editable ../" + p["path"])

    for p in comaintain_packages:
        print(p["name"])
