import streamlit as st
from firebase_admin import auth, firestore
from typing import Dict
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from google.cloud.firestore_v1 import FieldFilter


@lru_cache(maxsize=1)
def get_firestore_client():
    """Initialize and return Firestore client."""
    try:
        return firestore.client()
    except ValueError as e:
        st.error(f"Firestore initialization error: {str(e)}. Ensure Firebase is initialized.")
        raise

def ensure_parent_document_exists(db, collection_name, user_id):
    """Ensure the parent document exists for a given user_id in the specified collection."""
    doc_ref = db.collection(collection_name).document(user_id)
    if not doc_ref.get().exists:
        doc_ref.set({})  # Create an empty document if it doesn't exist


def log_user_activity(user_id: str, action: str, details: Dict = {}):
    """Log user activity to Firestore for both user_logs and admin_logs."""
    try:
        db = get_firestore_client()
        ensure_parent_document_exists(db, "user_logs", user_id)
        log_entry = {
            "user_id": user_id,
            "email": st.session_state.user_info.get('email', 'Unknown') if st.session_state.get(
                'user_info') else 'Unknown',
            "action": action,
            "details": details,
            "timestamp": firestore.SERVER_TIMESTAMP
        }
        db.collection("user_logs").document(user_id).collection("logs").add(log_entry)
        db.collection("admin_logs").add(log_entry)
        print(f"Logged activity: {action} for user {user_id}, email: {log_entry['email']}")
    except Exception as e:
        st.error(f"Error logging activity: {str(e)}")
        log_error(user_id, f"Error logging activity: {str(e)}")


def store_feedback(user_id: str, feedback_entry: Dict):
    """Store feedback in Firestore, including star ratings."""
    try:
        db = get_firestore_client()
        db.collection("feedback").document(user_id).collection("entries").add({
            "question": feedback_entry["question"],
            "response": feedback_entry["response"],
            "language": feedback_entry["language"],
            "rating": feedback_entry["rating"],
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        log_user_activity(user_id, "feedback", {
            "question": feedback_entry["question"],
            "rating": feedback_entry["rating"]
        })
    except Exception as e:
        st.error(f"Error storing feedback: {str(e)}")
        log_error(user_id, f"Error storing feedback: {str(e)}")


def store_chat_history(user_id: str, chat_entry: Dict):
    """Store chat history in Firestore."""
    try:
        db = get_firestore_client()
        ensure_parent_document_exists(db, "chat_history", user_id)
        db.collection("chat_history").document(user_id).collection("chats").add({
            "query": chat_entry["query"],
            "response": chat_entry["response"],
            "language": chat_entry["language"],
            "type": chat_entry["type"],
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        log_user_activity(user_id, "chat", {
            "query": chat_entry["query"],
            "language": chat_entry["language"]
        })
    except Exception as e:
        st.error(f"Error storing chat: {str(e)}")
        log_error(user_id, f"Error storing chat: {str(e)}")


def set_admin_user(email: str):
    """Set a user as admin in Firebase with rate-limiting and error handling."""
    try:
        if not st.session_state.get('user_info'):
            st.error("You must be logged in to perform this action.")
            return
        user_id = st.session_state.user_info['localId']

        # Rate-limiting: Limit to 5 admin actions per user per day
        db = get_firestore_client()
        rate_limit_timestamp = datetime.now(timezone.utc) - timedelta(days=1)
        recent_actions = db.collection("admin_logs") \
            .where(filter=FieldFilter("user_id", "==", user_id)) \
            .where(filter=FieldFilter("action", "==", "grant_admin")) \
            .where(filter=FieldFilter("timestamp", ">=", rate_limit_timestamp)) \
            .count().get()
        if recent_actions[0][0].value >= 5:
            st.error("Too many admin actions in the last 24 hours. Please try again later.")
            return

        # Get user by email and verify email
        user = auth.get_user_by_email(email)
        if not user.email_verified:
            st.error("User's email must be verified before granting admin access.")
            return

        # Set admin claim
        auth.set_custom_user_claims(user.uid, {"admin": True})
        st.success(f"Admin access granted to {email}. They must log out and log back in.")
        log_user_activity(user_id, "grant_admin", {"email": email})
    except auth.UserNotFoundError:
        st.error("Error: User not found or invalid email.")
        log_error(user_id, f"User not found for email: {email}")
    except Exception as e:
        st.error(f"Error setting admin: {str(e)}")
        log_error(user_id, f"Error setting admin: {str(e)}")


def log_error(user_id: str, error_message: str):
    """Log errors to Firestore for debugging."""
    try:
        db = get_firestore_client()
        db.collection("error_logs").add({
            "user_id": user_id or "unknown",
            "error": error_message,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        print(f"Failed to log error: {str(e)}")