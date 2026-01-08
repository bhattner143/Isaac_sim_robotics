from onshape_robotics_toolkit.connect import Client
from onshape_robotics_toolkit.models.document import Document
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
key_env_path = os.path.join(script_dir, "key.env")

# Initialize the client
client = Client(
    env=key_env_path
)

# Create a Document object from a URL
doc = Document.from_url(
    url="https://cad.onshape.com/documents/02830e503a1c2626d58e0780/w/62db7934caa01effe231f1a9/e/9d31039ea010abe028749d07"
)

# Retrieve the assembly and its JSON representation
assembly = client.get_assembly(
    did=doc.did,
    wtype=doc.wtype,
    wid=doc.wid,
    eid=doc.eid
)

# Print assembly details in a nice format
print("=" * 80)
print(f"ASSEMBLY: {assembly.name}")
print("=" * 80)

print(f"\nDocument: {doc.name}")
print(f"Configuration: {assembly.rootAssembly.configuration}")

print(f"\n{'PARTS':-^80}")
print(f"Total parts: {len(assembly.parts)}")
for i, part in enumerate(assembly.parts, 1):
    print(f"\n{i}. Part ID: {part.partId}")
    print(f"   Body Type: {part.bodyType}")
    print(f"   Configuration: {part.configuration}")

print(f"\n{'INSTANCES':-^80}")
print(f"Total instances: {len(assembly.rootAssembly.instances)}")
for i, inst in enumerate(assembly.rootAssembly.instances, 1):
    print(f"\n{i}. {inst.name}")
    print(f"   Type: {inst.type.value}")
    print(f"   ID: {inst.id}")
    print(f"   Part ID: {inst.partId}")
    print(f"   Suppressed: {inst.suppressed}")

print(f"\n{'JOINTS/MATES':-^80}")
print(f"Total mates: {len(assembly.rootAssembly.features)}")
for i, feature in enumerate(assembly.rootAssembly.features, 1):
    if hasattr(feature, 'featureData'):
        print(f"\n{i}. {feature.featureData.name}")
        print(f"   Type: {feature.featureData.mateType.value}")
        print(f"   ID: {feature.id}")
        print(f"   Suppressed: {feature.suppressed}")
        print(f"   Connected parts:")
        for j, entity in enumerate(feature.featureData.matedEntities, 1):
            occurrence_name = assembly.rootAssembly.instances[[inst.id for inst in assembly.rootAssembly.instances].index(entity.matedOccurrence[0])].name if entity.matedOccurrence else "Unknown"
            print(f"      {j}. {occurrence_name}")
            print(f"         Origin: {entity.matedCS.origin}")

print(f"\n{'OCCURRENCES':-^80}")
print(f"Total occurrences: {len(assembly.rootAssembly.occurrences)}")
for i, occ in enumerate(assembly.rootAssembly.occurrences, 1):
    instance_name = assembly.rootAssembly.instances[[inst.id for inst in assembly.rootAssembly.instances].index(occ.path[0])].name if occ.path else "Unknown"
    print(f"\n{i}. {instance_name}")
    print(f"   Path: {occ.path}")
    print(f"   Fixed: {occ.fixed}")
    print(f"   Hidden: {occ.hidden}")
    print(f"   Transform: [4x4 matrix]")

print("\n" + "=" * 80)