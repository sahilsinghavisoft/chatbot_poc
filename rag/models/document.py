from mongoengine import connect, Document, StringField, ListField, FloatField,URLField

connect('your_database_name')

class TextDocument(Document):
    content = StringField(required=True)
    source_url = StringField()
    embedding = ListField(FloatField())

    meta = {
        'collection': 'documents'
    }

# Create index manually if needed
TextDocument._get_collection().create_index([('embedding', 1)], name='embedding_index')