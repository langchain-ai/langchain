class MyClass {
  constructor(name) {
    this.name = name;
  }

  greet() {
    console.log(`Hello, ${this.name}!`);
  }
}

function main() {
  const name = prompt("Enter your name:");
  const obj = new MyClass(name);
  obj.greet();
}

main();
