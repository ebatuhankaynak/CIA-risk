Export a project as a .jar file from IntelliJ IDEA as artifact

Static
javacg-static accepts as arguments the jars to analyze.

java -jar javacg-0.1-SNAPSHOT-static.jar lib1.jar lib2.jar... > output.txt
javacg-static produces combined output in the following format:

For methods
  M:class1:<method1>(arg_types) (typeofcall)class2:<method2>(arg_types)
The line means that method1 of class1 called method2 of class2. The type of call can have one of the following values (refer to the JVM specification for the meaning of the calls):

M for invokevirtual calls
I for invokeinterface calls
O for invokespecial calls
S for invokestatic calls
D for invokedynamic calls
For invokedynamic calls, it is not possible to infer the argument types.

For classes
  C:class1 class2
This means that some method(s) in class1 called some method(s) in class2.