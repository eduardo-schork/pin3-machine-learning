from datetime import datetime
import firebase_admin
from firebase_admin import credentials, storage, db, auth

cred = credentials.Certificate("./service-account-key.json")
firebase_admin.initialize_app(
    cred,
    {
        "storageBucket": "pin3-42a6f.appspot.com",
        "databaseURL": "https://pin3-42a6f-default-rtdb.firebaseio.com/",
    },
)


def get_request_auth_token(request):
    token = request.headers.get("Authorization").split(" ")[1]
    decoded_token = auth.verify_id_token(token)
    uid = decoded_token["uid"]
    return uid


def save_image_to_bucket(user_id, image_content, image_filename, image_type):
    bucket = storage.bucket()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    image_name = f"post-images/{timestamp}_{user_id}_{image_filename}"
    blob = bucket.blob(image_name)
    blob.upload_from_string(image_content, content_type=image_type)

    blob.make_public()

    url = blob.public_url

    return url


def save_post_to_db(
    user_id,
    image_url,
    feedback_class,
    predicted_class,
    predicted_percentage,
    model_type,
):
    ref = db.reference("tb_feed")
    new_post = ref.push(
        {
            "user_id": user_id,
            "image_url": image_url,
            "predicted_percentage": predicted_percentage,
            "predicted_class": predicted_class,
            "feedback_class": feedback_class,
            "model_type": model_type,
        }
    )
    return new_post


def get_posts_from_db():
    ref = db.reference("tb_feed")
    posts = ref.get()
    return posts
