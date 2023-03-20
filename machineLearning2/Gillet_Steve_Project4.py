from mvpa2.suite import *
filepath = os.path.join(pymvpa_datadbroot, 'mnist', "mnist.hdf5")
datasets = h5load(filepath)
train = datasets['train']
test = datasets['test']
print(train)

print(test)

# assign a mapper able to recreate 28x28 pixel image arrays
test.a.mapper = FlattenMapper(shape=(28, 28))
test.mapper.reverse(test).shape