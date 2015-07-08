<div>There use of Generics:</div><div><ul><li>Using a generic class, like using ArrayList&lt;String&gt;</li><li>Writing generic code with simple &lt;T&gt; or &lt;?&gt; type parameter.</li><li>Writing generic code with a &lt;T extends Foo&gt; type parameter.</li></ul></div>
<h1>Generic 1 — Use Generic Class</h1><div><ul><li>Many Java library classes have been made generic, so in stead of just raw Object, they can be used in a way that indicates the type of object they hold.</li></ul></div>
```java
ArrayList<String> string = new ArrayList<String>();
strings.add("Hi");
strings.add("there");
String s = strings.get(0); // no cast required
```
<div><ul><li>The plain types such as List are known as&nbsp;<b>raw version</b>&nbsp;as it still works in Java but they just contain pointers of type Object. Raw version and generic can be assigned back and fort with a IDE warning. &nbsp;</li></ul></div>
<h3>Boxing / Unboxing</h3><div><ul><li>Normally, it’s not possible to store int or boolean in an ArrayList since it can only store pointers to objects. ArrayList&lt;int&gt; should be ArrayList&lt;Integer&gt;</li><li>With Java 5 “auto boxing”, when the code needs an Integer but has an int, it automatically creates the Integer on the fly. “Auto unboxing” is to call intValue() on the fly when int is needed in place of an Integer (object).</li></ul><h3>Warning: Unboxing Does Not Work With == or !=</h3></div>
```java
List<Integer> a, b;     // two list of Integer a and b
a.get(0) == b.get(0);   // this just compare two references (pointers)
                        // Auto unboxing doesn't work here
a.get(0).equals(b.get(0));  // This will properly compares int values
// It works this way to remain compatible with the original definition of ==
```
<h3>Foreach Loop</h3>
```java
List<String> strings = ...

for (String s : strings)
```