The aim is to create an `IValue` from received data, to convert a data to IValue we have
from (Tensor), from (bool), from (long), from (double), from(String), from (bool List)
from (long List) (for int), from (double List) (for float), and also basically Tensor is used
for IValues that are list, if we see a list in `dataArg` then convert it to `Tensor`,
now Tensor can be created from
byte[] (torch.int8), int[] (torch.int32), float[] (torch.float32), long[] (torch.int64),
double[] (torch.flo
if dataArg is ArrayList instance, then convert it to Tensor of dtype,
     then create IValue out of it
else convert dataArg to its datatype and create an IValue out
dataArg can be of type Integer, Long, Double, String, Arr
so if it is an Integer, Long, Double or String then convert them to its respective IValue
or else if its an ArrayList, it can be of Integer, Long or Double, these will then be needed
to be converted to their desire
Generally speaking, these conversions are preferred, but you know people can can stuff like
pass in an ArrayList<Integer> and convert it to Tensor[INT64], so that is allowed.
BUT ArrayList<String> is NOT ALLOWED, fo
   List[Long] -> Tensor[INT64], Tensor[INT32]
   List[Integer] -> Tensor[INT32], Tensor[INT64], Tensor[INT8]
   List[Double] -> Tensor[FLOAT32], Tensor[FLOAT64]

the dataArg here can be of type
Boolean, Integer, Long, Double, String, byte[], int[], long[], double[]
*It could also be an ArrayList, but i've decided that i wont be supporting that for now*
its corresponding Flutter types are
bool, int, int,
out of these byte[], int[], long[], double[] can be converted to a `Tensor`
byte[] -> Tensor[INT8], Tensor[UINT8]
int[] -> Tensor[INT32]
long[] -> Tensor[INT64]
double[] -> Tensor[FLOAT64], Tensor[FLOAT32] (double[] has to be converted to float[])
When you receive the data, try to typecast it to the type defined in dtype
Something to note is that all the array types are primitive already, so there wont be
any issues of unboxing, but for non-array types we have to un-box them
But if the user sends in a List then it will be converted to ArrayList in Java, this is
the case only when the the user wants to create an IValue which is a list, but is not a
Tensor, in this case we have to un-box the array
you can use `instanceof` operator to check for the datatype of incoming dataArg Object