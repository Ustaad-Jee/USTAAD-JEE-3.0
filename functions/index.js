const functions = require('firebase-functions');
const admin = require('firebase-admin');
admin.initializeApp();

exports.setAdminRole = functions.https.onCall(async (data, context) => {
  if (!context.auth) {
    throw new functions.https.HttpsError('unauthenticated', 'Only authenticated users can set admin roles.');
  }
  if (!context.auth.token.admin) {
    throw new functions.https.HttpsError('permission-denied', 'Only admins can set admin roles.');
  }
  const uid = data.uid;
  if (!uid) {
    throw new functions.https.HttpsError('invalid-argument', 'UID is required.');
  }
  try {
    const user = await admin.auth().getUser(uid);
    if (!user.emailVerified) {
      throw new functions.https.HttpsError('failed-precondition', 'User email must be verified.');
    }
    await admin.auth().setCustomUserClaims(uid, { admin: true });
    // Log the admin action
    await admin.firestore().collection('admin_logs').add({
      user_id: context.auth.uid,
      email: context.auth.token.email || 'Unknown',
      action: 'grant_admin',
      details: { target_uid: uid, target_email: user.email || 'Unknown' },
      timestamp: admin.firestore.FieldValue.serverTimestamp()
    });
    return { message: `Admin role assigned to user with UID ${uid}` };
  } catch (error) {
    throw new functions.https.HttpsError('internal', `Error setting admin role: ${error.message}`);
  }
});
